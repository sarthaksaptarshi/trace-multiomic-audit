"""Non-circular patient-level validation suite (SIMULATED DATA).
External performance is measured from REAL predictions on held-out target
patients; TRACE diagnostics are computed from the FITTED pipeline and permitted
audit data. Mechanisms (batch, signal, prevalence, noise, missing modality,
unmeasured corruption) vary INDEPENDENTLY. Splits: source-train, source-val,
target-audit (small, labelled), target-eval (labelled; used only to score the
outcome). Unlabelled target features are available to label-free diagnostics
(transductive, disclosed)."""
import warnings, numpy as np; warnings.filterwarnings("ignore")
from numpy.random import default_rng
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import StratifiedKFold
RNG=default_rng(20); K=4
PB=[25,60,80,40,30]; MODS=[2,2,4,2,2]; MOD=7; P=sum(PB); METH0=PB[0]

def ece(p,y,M=10):
    c=p.max(1); pr=p.argmax(1); a=(pr==y); b=np.linspace(0,1,M+1); e=0
    for i in range(M):
        m=(c>b[i])&(c<=b[i+1])
        if m.sum(): e+=m.mean()*abs(a[m].mean()-c[m].mean())
    return e
def ent(p): return (-(p*np.log(p+1e-9)).sum(1)).mean()

def build_W(rng, amp):
    W=np.zeros((P,K)); col=0
    for b,cnt in enumerate(MODS):
        base=sum(PB[:b])
        for _ in range(cnt):
            W[base:base+MOD, col%K]+=rng.uniform(0.7,1.3)*amp; base+=MOD; col+=1
    return W

def deployment(rng, knobs=None):
    # independent mechanisms
    k = knobs or {}
    batch = k.get("batch", rng.uniform(0,1))
    amp   = k.get("amp",   rng.uniform(0.45,1.1))      # biological signal
    noise = k.get("noise", rng.uniform(0.9,1.4))
    prev  = k.get("prev",  rng.uniform(0,1))           # target prevalence shift
    miss  = k.get("miss",  rng.random()<0.25)          # drop a modality at target
    unmeas= k.get("unmeas",rng.uniform(0,1))           # target-only corruption (invisible)
    W=build_W(rng, amp)
    def sample(n, external, p_classes):
        y=rng.choice(K, n, p=p_classes)
        X=rng.standard_normal((n,P))*noise + W[:,y].T
        bf=slice(METH0,METH0+8)
        if external:
            X[:,bf]+=1.5*batch*(rng.permutation(np.eye(K))[y]@rng.standard_normal((K,8)))
            # unmeasured corruption: nonlinear squashing of a random feature block
            cb=slice(METH0+PB[1], METH0+PB[1]+20)
            X[:,cb]=X[:,cb]+unmeas*1.5*np.tanh(X[:,cb]*2)
            if miss:  # missing modality -> impute by source mean (0)
                X[:, sum(PB[:4]):]=0.0
        else:
            X[:,bf]+=1.5*batch*(np.eye(K)[y]@rng.standard_normal((K,8)))
        return X,y
    ps=np.ones(K)/K
    pt=rng.dirichlet(np.ones(K)*(0.4+3*(1-prev)))      # prev high -> skewed target
    Xs,ys=sample(360,False,ps)
    Xt,yt=sample(300,True, pt)
    # splits
    Xtr,ytr=Xs[:260],ys[:260]; Xsv,ysv=Xs[260:],ys[260:]
    Xau,yau=Xt[:40],yt[:40];   Xev,yev=Xt[40:],yt[40:]
    clf=LogisticRegression(max_iter=300,C=0.5).fit(Xtr,ytr)
    f1_ext=f1_score(yev,clf.predict(Xev),average="macro")   # OUTCOME (real preds)
    pev=clf.predict_proba(Xev); pau=clf.predict_proba(Xau); psv=clf.predict_proba(Xsv)
    # ---- label-dependent (target-audit labels) ----
    dECE=ece(pau,yau)-ece(psv,ysv)
    direct=f1_score(yau,clf.predict(Xau),average="macro")   # direct estimate of f1_ext
    # ---- label-free diagnostics (source + unlabelled target) ----
    conf_shift=psv.max(1).mean()-pau.max(1).mean()
    ent_shift =ent(pau)-ent(psv)
    # TRACE build-quality probes (source only)
    sel=np.argsort(f_classif(Xtr,ytr)[0])[-40:]
    bvals=[1-np.mean([np.var(Xtr[ytr==c,j][:max(2,(ytr==c).sum()//2)])/(np.var(Xtr[ytr==c,j])+1e-6) for c in range(K)]) for j in sel]
    qbar=float(np.clip(np.mean(bvals),0,1))
    h=rng.permutation(260); Sa=set(np.argsort(f_classif(Xtr[h[:130]],ytr[h[:130]])[0])[-40:]); Sb=set(np.argsort(f_classif(Xtr[h[130:]],ytr[h[130:]])[0])[-40:])
    stab=len(Sa&Sb)/len(Sa|Sb)
    # leakage susceptibility: permuted-label CV, selection outside vs inside
    yp=rng.permutation(ytr)
    def cvacc(outside):
        acc=[]
        for tr,te in StratifiedKFold(3,shuffle=True,random_state=1).split(Xtr,yp):
            fe=np.argsort(f_classif(Xtr,yp)[0])[-25:] if outside else np.argsort(f_classif(Xtr[tr],yp[tr])[0])[-25:]
            acc.append((LogisticRegression(max_iter=150).fit(Xtr[tr][:,fe],yp[tr]).predict(Xtr[te][:,fe])==yp[te]).mean())
        return np.mean(acc)
    leak=max(0,cvacc(True)-cvacc(False))
    # label-free baselines on unlabelled target (eval features, no labels)
    conf=pev.max(1).mean(); entropy=ent(pev)
    def sboot(): return np.concatenate([rng.choice(np.where(ytr==c)[0],(ytr==c).sum(),replace=True) for c in range(K)])
    ens=np.stack([LogisticRegression(max_iter=200,C=0.5).fit(Xtr[(i:=sboot())],ytr[i]).predict_proba(Xev) for _ in range(5)]).var(0).mean()
    Xd=np.vstack([Xtr,Xev]); yd=np.r_[np.zeros(260),np.ones(len(Xev))]
    dom=np.mean([roc_auc_score(yd[te],LogisticRegression(max_iter=200).fit(Xd[tr],yd[tr]).predict_proba(Xd[te])[:,1]) for tr,te in StratifiedKFold(3,shuffle=True,random_state=0).split(Xd,yd)])
    return dict(f1=f1_ext,dECE=dECE,direct=direct,conf_shift=conf_shift,ent_shift=ent_shift,
                qbar=qbar,stab=stab,leak=leak,conf=conf,entropy=entropy,ens=ens,dom=dom,
                yev=yev, pev=pev)

RETRIES={"calls":0,"redrawn":0,"errors":{}}
def safe(rng,**kw):
    """Draw a deployment; if a draw fails (e.g. degenerate class split), redraw.
    Redraws are counted and reported (retry_report) so any selection is disclosed."""
    RETRIES["calls"]+=1
    for _ in range(6):
        try: return deployment(rng,**kw)
        except Exception as e:
            RETRIES["redrawn"]+=1; k=type(e).__name__; RETRIES["errors"][k]=RETRIES["errors"].get(k,0)+1
    raise RuntimeError("deployment failed 6 times in a row")
def retry_report():
    print("  [disclosure] deployments requested %d, failed draws redrawn %d %s"
          %(RETRIES["calls"],RETRIES["redrawn"],RETRIES["errors"] or ""))
if __name__=='__main__':
    import os; os.makedirs("figures",exist_ok=True)
    M=180
    rows=[safe(RNG) for _ in range(M)]
    A={k:np.array([r[k] for r in rows],dtype=object) if k in ("yev","pev") else np.array([r[k] for r in rows]) for k in rows[0]}
    def ci(fn,*xs,B=1500):
        """point estimate on the full sample + percentile bootstrap 95% CI"""
        n=len(xs[0]); s=[]
        for _ in range(B):
            i=RNG.integers(0,n,n)
            try:s.append(fn(*[x[i] for x in xs]))
            except Exception:pass
        return fn(*xs),np.nanpercentile(s,2.5),np.nanpercentile(s,97.5)

    print("NON-CIRCULAR PATIENT-LEVEL VALIDATION  (M=%d deployments)"%M)
    print("external macro-F1: min %.2f median %.2f max %.2f"%(A["f1"].min(),np.median(A["f1"]),A["f1"].max()))
    sp=M//2; dz=slice(0,sp); ez=slice(sp,M)
    def mm(x,ref): lo,hi=ref.min(),ref.max(); return np.clip((x-lo)/(hi-lo+1e-9),0,1)
    Tlab=np.mean([mm(A["qbar"],A["qbar"][dz]),A["stab"],1-mm(A["leak"],A["leak"][dz]),1-mm(np.abs(A["dECE"]),np.abs(A["dECE"])[dz])],0)
    Tfree=np.mean([mm(A["qbar"],A["qbar"][dz]),A["stab"],1-mm(A["leak"],A["leak"][dz]),1-mm(np.abs(A["ent_shift"]),np.abs(A["ent_shift"])[dz])],0)
    BAR=float(np.median(A["f1"][dz])); fail=(A["f1"]<BAR).astype(int)
    print("design-split failure bar %.2f ; eval prevalence %.2f"%(BAR,fail[ez].mean()))
    print("\nPredicting external macro-F1 (eval split): Pearson r [CI]  |  failure AUROC [CI]  |  AUPRC [CI]   (label access)")
    detectors=[("direct estimate (target-audit labels)",A["direct"],"LABELS"),
               ("calibration drift (audit labels)",-np.abs(A["dECE"]),"LABELS"),
               ("provenance (source)",mm(A["qbar"],A["qbar"][dz]),"free"),
               ("stability (source)",A["stab"],"free"),
               ("leakage (source)",1-mm(A["leak"],A["leak"][dz]),"free"),
               ("predictive confidence",A["conf"],"free"),
               ("predictive entropy",-A["entropy"],"free"),
               ("ensemble disagreement",-A["ens"],"free"),
               ("domain classifier",-A["dom"],"free"),
               ("TRACE label-dependent",Tlab,"LABELS"),
               ("TRACE label-free",Tfree,"free")]
    out={}
    for nm,sc,acc in detectors:
        r=ci(lambda a,b:pearsonr(a,b)[0],sc[ez],A["f1"][ez])
        au=ci(lambda y,s:roc_auc_score(y,-s),fail[ez],sc[ez])
        ap=ci(lambda y,s:average_precision_score(y,-s),fail[ez],sc[ez])
        print("  %-38s r=%.2f[%.2f,%.2f]  AUROC=%.2f[%.2f,%.2f]  AUPRC=%.2f  (%s)"%(nm,*r,*au,ap[0],acc))
        out[nm]=dict(r=r,auroc=au,auprc=ap,access=acc)
    import results_io; results_io.save("comprehensive",out)
    print("  base-rate AUPRC = %.2f"%fail[ez].mean())
    np.savez("figures/comp.npz", f1=A["f1"], Tfree=Tfree, Tlab=Tlab, direct=A["direct"],
             dom=A["dom"], fail=fail, dECE=A["dECE"], ent_shift=A["ent_shift"])
    print("saved comp.npz"); retry_report()