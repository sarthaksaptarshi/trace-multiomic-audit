import warnings,numpy as np; warnings.filterwarnings("ignore")
from numpy.random import default_rng
from sklearn.metrics import f1_score
from comprehensive import deployment, safe, ece, ent
RNG=default_rng(5)

# ---------- 2. DEFERRAL ----------
print("=== DEFERRAL ===")
# (a) within-deployment: scaling the confidence threshold by a constant T cannot
#     change ranking -> identical risk-coverage. Demonstrate on one deployment.
d=safe(RNG); pev=d["pev"]; yev=np.array(d["yev"]); conf=pev.max(1); correct=(pev.argmax(1)==yev)
def risk_cov(score, cov):
    k=int(len(score)*cov); idx=np.argsort(-score)[:k]; return 1-correct[idx].mean()
cov_grid=[1.0,0.8,0.6,0.4]
rc_conf=[risk_cov(conf,c) for c in cov_grid]
rc_scaled=[risk_cov(conf*0.3,c) for c in cov_grid]   # T=0.3 constant scaling
print("  within-deployment risk at coverage",cov_grid)
print("   confidence-only   :",[f"{x:.3f}" for x in rc_conf])
print("   trust-scaled thr. :",[f"{x:.3f}" for x in rc_scaled],"(identical -> no gain)")
# (b) across-deployment gating: defer whole low-trust deployments
RNG2=default_rng(9); M=80; rows=[safe(RNG2) for _ in range(M)]
f1=np.array([r["f1"] for r in rows])
err=np.array([1-f1_score(np.array(r["yev"]),r["pev"].argmax(1),average="micro") for r in rows])  # patient error rate
npat=np.array([len(r["yev"]) for r in rows])
def mm(x): return (x-x.min())/(x.max()-x.min()+1e-9)
# trust (label-free) per deployment, recomputed from saved fields
Tf=np.mean([mm(np.array([r["qbar"] for r in rows])),
            np.array([r["stab"] for r in rows]),
            1-mm(np.array([r["leak"] for r in rows])),
            1-mm(np.abs(np.array([r["ent_shift"] for r in rows])))],0)
base_err=np.average(err,weights=npat)
print("\n  across-deployment gating (defer lowest-trust deployments):")
print("   baseline weighted error (all deployed): %.3f"%base_err)
for frac in [0.2,0.4]:
    keepT=np.argsort(-Tf)[:int(M*(1-frac))]; keepC=np.argsort(-np.array([r["conf"] for r in rows]))[:int(M*(1-frac))]
    eT=np.average(err[keepT],weights=npat[keepT]); eC=np.average(err[keepC],weights=npat[keepC])
    wl=1-npat[keepT].sum()/npat.sum()
    print("   defer %d%%: gate-by-TRACE err %.3f (abs red %.3f, rel %.0f%%); gate-by-confidence err %.3f ; deferred workload %.0f%%"
          %(int(frac*100),eT,base_err-eT,100*(base_err-eT)/base_err,eC,100*wl))

# ---------- 4. LEAKAGE: leaky vs correct pipeline ----------
print("\n=== LEAKAGE DIAGNOSTIC (permuted-label susceptibility) ===")
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
def leak_gap(X,y,rng,outside):
    yp=rng.permutation(y); acc=[]
    for tr,te in StratifiedKFold(3,shuffle=True,random_state=1).split(X,yp):
        fe=np.argsort(f_classif(X,yp)[0])[-25:] if outside else np.argsort(f_classif(X[tr],yp[tr])[0])[-25:]
        acc.append((LogisticRegression(max_iter=150).fit(X[tr][:,fe],yp[tr]).predict(X[te][:,fe])==yp[te]).mean())
    base=1.0/len(np.unique(y)); return np.mean(acc)-base
RNG3=default_rng(1); gi=[]; go=[]
for _ in range(30):
    d=safe(RNG3)  # reuse generator to get a source cohort
    # rebuild a small source cohort quickly from deployment internals is not exposed;
    # instead synthesise a quick labelled set with signal
    n=240; P=120; y=RNG3.integers(0,4,n); X=RNG3.standard_normal((n,P))+ (np.eye(4)[y]@RNG3.standard_normal((4,P)))*0.6
    gi.append(leak_gap(X,y,RNG3,False)); go.append(leak_gap(X,y,RNG3,True))
gi=np.array(gi); go=np.array(go)
print("  correct pipeline (in-fold selection):  permuted-label gap  mean %.3f (false-alarm if >0.1: %.0f%%)"%(gi.mean(),100*(gi>0.1).mean()))
print("  leaky  pipeline (outside-CV selection): permuted-label gap  mean %.3f (detected if >0.1: %.0f%%)"%(go.mean(),100*(go>0.1).mean()))

# ---------- 3. AUDIT-SIZE sensitivity (direct estimate) ----------
print("\n=== AUDIT-SIZE SENSITIVITY (direct F1 estimate vs audit n) ===")
from scipy.stats import pearsonr
for na in [20,40,80,160]:
    RNG4=default_rng(na); est=[]; tru=[]
    for _ in range(60):
        d=safe(RNG4); # use target-eval to form true; resample audit of size na from eval
        yev=np.array(d["yev"]); pev=d["pev"]
        idx=RNG4.choice(len(yev),min(na,len(yev)),replace=False)
        est.append(f1_score(yev[idx],pev[idx].argmax(1),average="macro")); tru.append(d["f1"])
    r=pearsonr(est,tru)[0]; print("  audit n=%3d : corr(direct estimate, true external F1) = %.2f"%(na,r))

# ---------- 3/6. SURROGATE COUNTEREXAMPLE ----------
print("\n=== SURROGATE COUNTEREXAMPLE (entropy shift vs true calibration drift) ===")
# regime where accuracy collapses but confidence stays high (confident errors):
# entropy shift ~ 0 while true dECE large -> surrogate should miss it.
RNG5=default_rng(7); es=[]; de=[]
for _ in range(80):
    n=200; P=80; y=RNG5.integers(0,4,n)
    Xs=RNG5.standard_normal((n,P))+(np.eye(4)[y]@RNG5.standard_normal((4,P)))*0.8
    from sklearn.linear_model import LogisticRegression as LR
    clf=LR(max_iter=200).fit(Xs,y)
    # target: relabel (confident but wrong) -> high ECE, confidence unchanged
    yt=(y+RNG5.integers(1,4,n))%4  # systematically wrong labels
    pt=clf.predict_proba(Xs)
    es.append(ent(pt)-ent(clf.predict_proba(Xs)))      # ~0 (same inputs)
    de.append(ece(pt,yt)-ece(clf.predict_proba(Xs),y))
from scipy.stats import pearsonr
print("  confident-error regime: mean entropy shift %.3f, mean true dECE %.3f, corr %.2f"%(np.mean(es),np.mean(de),pearsonr(es,de)[0]))
print("  -> entropy shift ~0 while calibration drift is large: surrogate fails here (counterexample).")
