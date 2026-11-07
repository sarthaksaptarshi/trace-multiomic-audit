"""Adversarial robustness / circularity stress test for the TRACE trust score.
All data are SIMULATED. Failure = external macro-F1 < prespecified bar (0.80).
Failure-detection AUROC uses each indicator in its intended 'higher = more
trustworthy' direction (score for the failure class = -indicator)."""
import zlib, numpy as np
from scipy.stats import pearsonr, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
BAR, B = 0.80, 2000
PROBES=["leakage","provenance","stability","calibration"]

def corr_uniform(n,k,rho,rng):
    cov=(1-rho)*np.eye(k)+rho*np.ones((k,k)); L=np.linalg.cholesky(cov)
    return norm.cdf(rng.standard_normal((n,k))@L.T)

def make_pool(n,rho,nonlinear,pnoise,omit_coef,lin_coef,rng):
    h=corr_uniform(n,4,rho,rng); h0,h1,h2,h3=h.T
    h5=rng.uniform(0,1,n) if omit_coef>0 else np.zeros(n)
    ext=0.95-lin_coef*h.mean(1)
    if nonlinear: ext=ext-0.12*(h0*h1+h2*h3)+0.06*h.mean(1)**2
    ext=ext-omit_coef*h5
    ext=(ext+rng.normal(0,0.02,n)).clip(0.20,0.97)
    P=np.column_stack([1-(h0+rng.normal(0,pnoise,n)).clip(0,1),   # leakage check
                       (1-h1+rng.normal(0,pnoise,n)).clip(0,1),   # provenance
                       (1-h2+rng.normal(0,pnoise,n)).clip(0,1),   # stability
                       1-(h3+rng.normal(0,pnoise,n)).clip(0,1)])  # calibration
    T=P.mean(1)
    intAcc=(0.90+0.06*h0+0.05*h1+rng.normal(0,0.03,n)).clip(0.6,0.995)
    return dict(ext=ext,P=P,T=T,intAcc=intAcc,fail=(ext<BAR).astype(int),h5=h5)

def fauroc(y,good_high):
    return roc_auc_score(y,-good_high) if len(np.unique(y))>1 else np.nan
def boot(func,*arr,rng=None):
    n=len(arr[0]); out=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        try:out.append(func(*[a[i] for a in arr]))
        except Exception:pass
    return func(*arr),np.nanpercentile(out,2.5),np.nanpercentile(out,97.5)   # point estimate + 95% CI
def f(t):return f"{t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}]"

def run(name,rho,nonlinear,pnoise,omit_coef,lin_coef):
    # deterministic per-scenario seed (Python's hash() of a str is randomised per process)
    rng=np.random.default_rng(zlib.crc32(name.encode("utf-8")))
    d=make_pool(400,rho,nonlinear,pnoise,omit_coef,lin_coef,rng)
    e=make_pool(400,rho,nonlinear,pnoise,omit_coef,lin_coef,rng)
    lr=LogisticRegression(max_iter=1000).fit(d["P"],d["fail"])
    Tl=lr.predict_proba(e["P"])[:,1]; y=e["fail"]
    r={"name":name,"prev":y.mean()}
    r["Teq"]=boot(fauroc,y,e["T"],rng=rng)
    r["Tlearn"]=boot(lambda yy,pp:roc_auc_score(yy,pp) if len(np.unique(yy))>1 else np.nan,y,Tl,rng=rng)
    r["rT"]=boot(lambda a,b:pearsonr(a,b)[0],e["T"],e["ext"],rng=rng)
    r["intacc_naive"]=boot(fauroc,y,e["intAcc"],rng=rng)   # naive 'high acc=trust'
    r["r_intacc"]=boot(lambda a,b:pearsonr(a,b)[0],e["intAcc"],e["ext"],rng=rng)
    aucs=[fauroc(y,e["P"][:,i]) for i in range(4)]; bi=int(np.nanargmax(aucs))
    r["best"]=(PROBES[bi],)+boot(fauroc,y,e["P"][:,bi],rng=rng)
    r["delta"]=boot(lambda yy,tt,ss:fauroc(yy,tt)-fauroc(yy,ss),y,e["T"],e["P"][:,bi],rng=rng)
    r["singles"]={PROBES[i]:aucs[i] for i in range(4)}
    if omit_coef>0:
        fi=np.where(y==1)[0]; r["omit_frac"]=(e["h5"][fi]>0.6).mean()
    return r

scen=[
 ("S0 idealized (independent, linear, clean)",      0.0,False,0.02,0.00,0.28),
 ("S1 correlated hazards",                          0.5,False,0.02,0.00,0.28),
 ("S2 correlated+nonlinear+noisy probes",           0.5,True, 0.15,0.00,0.28),
 ("S3 S2 + omitted mechanism (minor driver)",       0.5,True, 0.15,0.14,0.28),
 ("S4 omitted mechanism DOMINATES",                 0.5,True, 0.15,0.45,0.14),
 ("S5 heavy probe noise (sd=0.30)",                 0.5,True, 0.30,0.00,0.28),
]
print(f"Failure bar: external macro-F1 < {BAR} (prespecified). unit=configuration, "
      f"design=400/eval=400, bootstrap B={B}. Failure-detection AUROC in intended direction.\n")
ALL=[]
for s in scen:
    r=run(*s); ALL.append(r); print("="*76); print(r["name"]+f"   | failure prevalence {r['prev']:.2f}")
    print(f"  Trust T (equal)     failure-AUROC {f(r['Teq'])}   Pearson r(T,ext) {f(r['rT'])}")
    print(f"  Trust T (learned)   failure-AUROC {f(r['Tlearn'])}")
    print(f"  best single probe ({r['best'][0]})  failure-AUROC {r['best'][1]:.2f} [{r['best'][2]:.2f},{r['best'][3]:.2f}]")
    print(f"  Delta AUROC (T_equal - best single) {f(r['delta'])}")
    print(f"  internal accuracy (naive) failure-AUROC {f(r['intacc_naive'])}   Pearson r(intAcc,ext) {f(r['r_intacc'])}")
    print("  single-probe AUROCs: "+", ".join(f"{k} {v:.2f}" for k,v in r['singles'].items()))
    if 'omit_frac' in r: print(f"  >> {r['omit_frac']*100:.0f}% of failures mainly driven by the OMITTED hazard (invisible to TRACE)")
import results_io; results_io.save("stress_test",ALL)
