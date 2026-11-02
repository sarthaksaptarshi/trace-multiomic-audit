"""Does combining build-quality auditing with uncertainty baselines help?
EXP1: on the realistic (shift-driven) non-circular deployments.
EXP2: a disclosed, targeted regime with confident-error / unrepresentative-audit
      failures, where uncertainty and direct estimation are expected to fail and
      build-quality (provenance) may add value. All simulated."""
import warnings,numpy as np; warnings.filterwarnings("ignore")
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import pearsonr
from comprehensive import deployment, safe
RNG=default_rng(31)
def auroc(y,s): return roc_auc_score(y,s) if len(np.unique(y))>1 else np.nan
def ci(fn,*xs,B=1500,rng=RNG):
    n=len(xs[0]);out=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        try:out.append(fn(*[x[i] for x in xs]))
        except:pass
    return np.nanmean(out),np.nanpercentile(out,2.5),np.nanpercentile(out,97.5)

print("="*70,"\nEXP1: combination on realistic (shift-driven) non-circular data\n","="*70)
M=180; rows=[safe(RNG) for _ in range(M)]
G=lambda k:np.array([r[k] for r in rows])
f1=G("f1"); sp=M//2; dz=slice(0,sp); ez=slice(sp,M)
bar=np.median(f1[dz]); fail=(f1<bar).astype(int)
feats={"conf":-G("conf"),"entropy":-G("entropy"),"ens":-G("ens"),"dom":-G("dom"),
       "prov":-G("qbar"),"stab":-G("stab"),"leak":G("leak")}  # oriented high=fail
def fit_combo(keys):
    X=np.column_stack([feats[k] for k in keys])
    lr=LogisticRegression(max_iter=500).fit(X[dz],fail[dz]); return lr.predict_proba(X[ez])[:,1]
sets={"confidence only":["conf"],
      "uncertainty (conf+ent+ens+dom)":["conf","entropy","ens","dom"],
      "build-quality (prov+stab+leak)":["prov","stab","leak"],
      "combined (uncertainty+build-quality)":["conf","entropy","ens","dom","prov","stab","leak"]}
res={}
for nm,ks in sets.items():
    sc=fit_combo(ks); res[nm]=sc
    print("  %-40s AUROC %.2f [%.2f,%.2f]"%(nm,*ci(auroc,fail[ez],sc)))
# paired: combined - uncertainty
d=ci(lambda y,a,b:auroc(y,a)-auroc(y,b),fail[ez],res["combined (uncertainty+build-quality)"],res["uncertainty (conf+ent+ens+dom)"])
print("  Δ(combined − uncertainty) AUROC: %.3f [%.3f,%.3f]"%d)

print("\n"+"="*70,"\nEXP2: targeted confident-error / unrepresentative-audit regime\n","="*70)
print("  (disclosed hypothesis test: a regime built to probe whether build-quality")
print("   complements uncertainty; failures are confident and audit is misleading)")
def gen2(rng, build):
    P=120; K=4; ns=300; amp=0.35
    W=np.zeros((P,K))
    for k in range(K): W[k*12:(k+1)*12,k]=rng.uniform(0.6,1.0)*amp   # weak base signal
    sc=slice(60,75)                                                   # shortcut block
    def mk(n, sc_sign, shiftv):
        y=rng.integers(0,K,n); X=rng.standard_normal((n,P))+W[:,y].T
        batch=rng.standard_normal(n)
        # shortcut: strong class signal (sc_sign) + batch contamination (low provenance)
        X[:,sc]+= sc_sign*1.3*np.eye(K)[y]@rng.standard_normal((K,15)) + 0.9*batch[:,None]
        X+= shiftv*rng.standard_normal(P)
        return X,y
    if build:   # shortcut helps source+audit, flips on eval; little covariate shift
        Xs,ys=mk(ns,+1,0.0); Xa,ya=mk(60,+1,0.0); Xe,ye=mk(240,-1,0.0)
    else:       # shift regime: shortcut consistent, covariate shift on audit+eval
        sh=rng.uniform(0.6,1.3); Xs,ys=mk(ns,+1,0.0); Xa,ya=mk(60,+1,sh); Xe,ye=mk(240,+1,sh)
    clf=LogisticRegression(max_iter=300,C=0.5).fit(Xs,ys)
    f1e=f1_score(ye,clf.predict(Xe),average="macro")
    conf=clf.predict_proba(Xe).max(1).mean()
    direct=f1_score(ya,clf.predict(Xa),average="macro")
    sel=np.argsort(f_classif(Xs,ys)[0])[-40:]
    prov=float(np.clip(np.mean([1-np.mean([np.var(Xs[ys==c,j][:max(2,(ys==c).sum()//2)])/(np.var(Xs[ys==c,j])+1e-6) for c in range(K)]) for j in sel]),0,1))
    return f1e,conf,direct,prov
RNG2=default_rng(4); M2=200; D=[]
for _ in range(M2):
    b = RNG2.random()<0.5
    D.append(gen2(RNG2,b))
D=np.array(D); f1e,conf,direct,prov=D.T
sp=M2//2; dz=slice(0,sp); ez=slice(sp,M2)
bar2=np.median(f1e[dz]); fail2=(f1e<bar2).astype(int)
print("  eval failure prevalence %.2f ; external F1 median %.2f"%(fail2[ez].mean(),np.median(f1e)))
dets={"confidence":-conf,"direct estimate (audit)":-direct,"provenance":-prov}
for nm,sc in dets.items(): print("  %-26s AUROC %.2f [%.2f,%.2f]"%(nm,*ci(auroc,fail2[ez],sc)))
# combined confidence+provenance (fit on design)
X=np.column_stack([-conf,-prov]); lr=LogisticRegression(max_iter=500).fit(X[dz],fail2[dz]); cm=lr.predict_proba(X[ez])[:,1]
print("  %-26s AUROC %.2f [%.2f,%.2f]"%("combined (conf+prov)",*ci(auroc,fail2[ez],cm)))
dd=ci(lambda y,a,b:auroc(y,a)-auroc(y,b),fail2[ez],cm,-conf[ez])
print("  Δ(combined − confidence) AUROC: %.3f [%.3f,%.3f]"%dd)
