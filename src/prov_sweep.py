"""Item 3: provenance-misspecification sensitivity (config-level, correlated
moderate regime). Vary the noise with which the provenance diagnostic observes
its hazard; report trust-score failure-AUROC and r(T,ext) with bootstrap CIs."""
import numpy as np
from scipy.stats import pearsonr, norm
from sklearn.metrics import roc_auc_score
RNG=np.random.default_rng(3); BAR=0.80; B=1500
def corr_u(n,k,rho,rng):
    cov=(1-rho)*np.eye(k)+rho*np.ones((k,k)); L=np.linalg.cholesky(cov)
    return norm.cdf(rng.standard_normal((n,k))@L.T)
def pool(n,qnoise,rng,rho=0.5,pn=0.15):
    h=corr_u(n,4,rho,rng); h0,h1,h2,h3=h.T
    ext=(0.95-0.28*h.mean(1)-0.12*(h0*h1+h2*h3)+0.06*h.mean(1)**2+rng.normal(0,0.02,n)).clip(0.2,0.97)
    Lh=(h0+rng.normal(0,pn,n)).clip(0,1)
    Q =(1-h1+rng.normal(0,qnoise,n)).clip(0,1)        # provenance w/ variable noise
    S =(1-h2+rng.normal(0,pn,n)).clip(0,1)
    Dc=(h3+rng.normal(0,pn,n)).clip(0,1)
    T=np.mean([1-Lh,Q,S,1-Dc],0); return T,ext
def ci(fn,*xs):
    n=len(xs[0]); s=[]
    for _ in range(B):
        i=RNG.integers(0,n,n)
        try:s.append(fn(*[x[i] for x in xs]))
        except Exception:pass
    return fn(*xs),np.nanpercentile(s,2.5),np.nanpercentile(s,97.5)   # point estimate + 95% CI
print("provenance noise -> trust-score failure-AUROC and r(T,ext) [95% CI]")
print("%8s %22s %22s"%("sd(q)","failure-AUROC","Pearson r(T,ext)"))
OUT=[]
for qn in [0.0,0.1,0.2,0.3,0.4,0.5]:
    T,ext=pool(400,qn,RNG); fail=(ext<BAR).astype(int)
    au=ci(lambda y,s:roc_auc_score(y,-s),fail,T); r=ci(lambda a,b:pearsonr(a,b)[0],T,ext)
    print("%8.2f  %.2f [%.2f,%.2f]        %.2f [%.2f,%.2f]"%(qn,*au,*r))
    OUT.append(dict(qnoise=qn,auroc=au,r=r))
import results_io; results_io.save("prov_sweep",OUT)
