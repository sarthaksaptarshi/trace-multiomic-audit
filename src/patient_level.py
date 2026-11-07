"""Patient-level deployment simulator (SIMULATED DATA).
Item 1: validate a LABEL-FREE calibration-drift surrogate (calibration-varying
        regime, accuracy stable).
Item 2: failure-detection baselines vs a fully label-free trust score
        (accuracy-failing regime), AUROC + AUPRC, design-split-prespecified bar.
Unit = deployment; bootstrap CIs over deployments."""
import warnings, numpy as np; warnings.filterwarnings("ignore")
from numpy.random import default_rng
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
RNG=default_rng(7); K=4; PB=[30,80,100,50,40]; MOD=8; P=sum(PB); METH0=PB[0]

def ece(p,y,M=10):
    c=p.max(1); pr=p.argmax(1); a=(pr==y); b=np.linspace(0,1,M+1); e=0
    for i in range(M):
        m=(c>b[i])&(c<=b[i+1])
        if m.sum(): e+=m.mean()*abs(a[m].mean()-c[m].mean())
    return e
def temper(p,T): q=np.clip(p,1e-6,1)**(1/T); return q/q.sum(1,keepdims=True)
def ent(p): return (-(p*np.log(p+1e-9)).sum(1)).mean()

def gen(rng, sig, shift, batch, nch):
    s_dom=rng.uniform(0,1); s_b=rng.uniform(0,1); n=int(rng.choice(nch)); s_cal=rng.uniform(0,1)
    W=np.zeros((P,K)); mods=[2,2,4,2,2]; col=0
    for bb,cnt in enumerate(mods):
        base=sum(PB[:bb])
        for _ in range(cnt):
            W[base:base+MOD, col%K]+=rng.uniform(*sig); base+=MOD; col+=1
    def samp(nn,ext):
        y=rng.integers(0,K,nn); X=rng.standard_normal((nn,P))*0.9+W[:,y].T
        if ext: X+=s_dom*rng.standard_normal(P)*shift
        bf=slice(METH0,METH0+10)
        if ext: X[:,bf]+=batch*s_b*(rng.permutation(np.eye(K))[y]@rng.standard_normal((K,10)))
        else:   X[:,bf]+=batch*s_b*(np.eye(K)[y]@rng.standard_normal((K,10)))
        return X,y
    Xtr,ytr=samp(n,False); Xte,yte=samp(200,False); Xex,yex=samp(200,True)
    clf=LogisticRegression(max_iter=300,C=0.5).fit(Xtr,ytr)
    f1=f1_score(yex,clf.predict(Xex),average="macro")
    pin=clf.predict_proba(Xte); pex=temper(clf.predict_proba(Xex),0.6+1.8*s_cal)
    dECE=ece(pex,yex)-ece(pin,yte)
    conf_shift=pin.max(1).mean()-pex.max(1).mean(); ent_shift=ent(pex)-ent(pin)
    hi,_=np.histogram(pin.max(1),20,(0,1),density=True); he,_=np.histogram(pex.max(1),20,(0,1),density=True)
    js=jensenshannon(hi+1e-9,he+1e-9)
    # baselines
    def sboot():
        idx=np.concatenate([rng.choice(np.where(ytr==k)[0],max(2,(ytr==k).sum()),replace=True) for k in range(K)]); return idx
    ensm=[LogisticRegression(max_iter=200,C=0.5).fit(Xtr[(ii:=sboot())],ytr[ii]) for _ in range(5)]
    pe=np.stack([m.predict_proba(Xex) for m in ensm]); ens=pe.var(0).mean()
    Xd=np.vstack([Xtr,Xex]); yd=np.r_[np.zeros(n),np.ones(200)]
    dom=np.mean([roc_auc_score(yd[te],LogisticRegression(max_iter=200).fit(Xd[tr],yd[tr]).predict_proba(Xd[te])[:,1])
                 for tr,te in StratifiedKFold(3,shuffle=True,random_state=0).split(Xd,yd)])
    # label-free probes
    Fs=np.argsort(f_classif(Xtr,ytr)[0])[-50:]
    bvals=[1-np.mean([np.var(Xtr[ytr==k,j][:max(2,(ytr==k).sum()//2)])/(np.var(Xtr[ytr==k,j])+1e-6) for k in range(K)]) for j in Fs]
    qbar=float(np.clip(np.mean(bvals),0,1))
    h=rng.permutation(n); Sa=set(np.argsort(f_classif(Xtr[h[:n//2]],ytr[h[:n//2]])[0])[-50:]); Sb=set(np.argsort(f_classif(Xtr[h[n//2:]],ytr[h[n//2:]])[0])[-50:])
    stab=len(Sa&Sb)/len(Sa|Sb)
    yp=rng.permutation(ytr); of=np.argsort(f_classif(Xtr,yp)[0])[-30:]
    def cv(ff):
        ac=[]
        for tr,te in StratifiedKFold(3,shuffle=True,random_state=1).split(Xtr,yp):
            fe=ff if ff is not None else np.argsort(f_classif(Xtr[tr],yp[tr])[0])[-30:]
            ac.append((LogisticRegression(max_iter=150).fit(Xtr[tr][:,fe],yp[tr]).predict(Xtr[te][:,fe])==yp[te]).mean())
        return np.mean(ac)
    leak=max(0,cv(of)-cv(None))
    return dict(f1=f1,dECE=dECE,conf_shift=conf_shift,ent_shift=ent_shift,js=js,
                conf=pex.max(1).mean(),entropy=ent(pex),ens=ens,dom=dom,qbar=qbar,stab=stab,leak=leak)

REDRAWN=[0]
def gen_safe(rng,**kw):
    """redraw a failed deployment (counted and reported at the end)"""
    for _ in range(8):
        try: return gen(rng,**kw)
        except Exception: REDRAWN[0]+=1
    raise RuntimeError('gen failed')
def pool(M,**kw):
    rows=[gen_safe(RNG,**kw) for _ in range(M)]; return {k:np.array([r[k] for r in rows]) for k in rows[0]}
def ci(fn,*xs,B=2000):
    """point estimate on the full sample + percentile bootstrap 95% CI"""
    n=len(xs[0]); s=[]
    for _ in range(B):
        i=RNG.integers(0,n,n)
        try: s.append(fn(*[x[i] for x in xs]))
        except Exception: pass
    return (fn(*xs),np.nanpercentile(s,2.5),np.nanpercentile(s,97.5))

print("=== ITEM 1: label-free calibration surrogate vs TRUE drift (calibration-varying regime) ===")
A=pool(160, sig=(0.7,1.1), shift=1.5, batch=1.1, nch=(50,100,160))
print("  accuracy stable: external macro-F1 median %.2f (IQR %.2f-%.2f)"%(np.median(A["f1"]),np.percentile(A["f1"],25),np.percentile(A["f1"],75)))
SAVE={"item1":{},"item2":{}}
for nm,k in [("confidence shift","conf_shift"),("entropy shift","ent_shift"),("Jensen-Shannon","js")]:
    r=ci(lambda a,b:pearsonr(a,b)[0],A[k],A["dECE"]); rs=ci(lambda a,b:spearmanr(a,b)[0],A[k],A["dECE"])
    print("  %-20s Pearson %.2f [%.2f,%.2f]  Spearman %.2f [%.2f,%.2f]"%(nm,*r,*rs))
    SAVE["item1"][k]=dict(pearson=r,spearman=rs)

print("\n=== ITEM 2: failure detection vs baselines (accuracy-failing regime) ===")
B2=pool(220, sig=(0.35,0.75), shift=1.8, batch=1.4, nch=(40,80,160))
f1=B2["f1"]; print("  external macro-F1 min %.2f median %.2f max %.2f"%(f1.min(),np.median(f1),f1.max()))
sp=len(f1)//2; dz=slice(0,sp); ez=slice(sp,len(f1))
BAR=float(np.median(f1[dz]))                        # prespecified on DESIGN split
fail=(f1<BAR).astype(int); print("  bar (design-split median) %.2f ; eval prevalence %.2f"%(BAR,fail[ez].mean()))
def mm(x,ref): lo,hi=ref.min(),ref.max(); return np.clip((x-lo)/(hi-lo+1e-9),0,1)
calib_free=-np.abs(B2["conf_shift"])
Tfree=np.mean([B2["qbar"],B2["stab"],1-mm(B2["leak"],B2["leak"][dz]),mm(calib_free,calib_free[dz])],axis=0)
cands={"predictive confidence":-B2["conf"],"predictive entropy":B2["entropy"],
       "ensemble disagreement":B2["ens"],"domain classifier":B2["dom"],
       "TRACE trust score (label-free)":-Tfree}
print("  %-32s %16s %16s"%("detector","AUROC","AUPRC"))
for nm,sc in cands.items():
    au=ci(lambda y,s:roc_auc_score(y,s),fail[ez],sc[ez]); ap=ci(lambda y,s:average_precision_score(y,s),fail[ez],sc[ez])
    print("  %-32s %.2f [%.2f,%.2f]  %.2f [%.2f,%.2f]"%(nm,*au,*ap))
    SAVE["item2"][nm]=dict(auroc=au,auprc=ap)
print("  base rate (AUPRC chance) %.2f"%fail[ez].mean())
import os, results_io; os.makedirs("figures",exist_ok=True); results_io.save("patient_level",SAVE)
print("  [disclosure] failed deployment draws redrawn: %d"%REDRAWN[0])
np.savez("figures/patient_data.npz", dECE=A["dECE"], conf_shift=A["conf_shift"], ent_shift=A["ent_shift"],
         f1=B2["f1"], Tfree=Tfree, fail=fail, dom=B2["dom"], entropy=B2["entropy"])
print("saved.")
