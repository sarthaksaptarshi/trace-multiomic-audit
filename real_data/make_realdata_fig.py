"""Real-data figure from a single-cohort results CSV (xena_validation.py output).
Usage: python real_data/make_realdata_fig.py [--csv real_data/results_TCGA_BRCA_fixed.csv]"""
import argparse, os, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
ap=argparse.ArgumentParser(); ap.add_argument("--csv",default="real_data/results_TCGA_BRCA_fixed.csv")
R=pd.read_csv(ap.parse_args().csv); os.makedirs("figures",exist_ok=True)
rng=np.random.default_rng(0)
# majority-class macro-F1 per site (binary): majority F1 = 2(1-m)/(2-m), minority 0
maj=np.median([(1-m)/(2-m) for m in R.minority_frac.values]) if "minority_frac" in R else np.nan
def boot(x,y,n=5000):
    rs=[]
    for _ in range(n):
        i=rng.integers(0,len(x),len(x))
        if np.std(x[i])<1e-12 or np.std(y[i])<1e-12: continue
        rs.append(spearmanr(x[i],y[i]).correlation)
    rs=np.array(rs); return (np.nanpercentile(rs,2.5),np.nanpercentile(rs,97.5)) if len(rs) else (np.nan,np.nan)
orient=[("direct",1,"Direct estimate (audit labels)"),("confidence",1,"Predictive confidence"),
        ("entropy",-1,"Predictive entropy"),("stability",1,"Stability (source)"),
        ("dECE",-1,"Calibration drift (audit labels)"),("ensemble",-1,"Ensemble disagreement"),
        ("domain",-1,"Domain classifier"),("provenance",1,"Provenance (source)"),
        ("leakage",-1,"Leakage (source)")]
y=R.external_F1.values
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.6))
# Panel A: per-site external F1
o=np.argsort(y); sites=R.site.values[o]; f=y[o]; ne=R.n_eval.values[o]
ax1.barh(range(len(f)),f,color="#4C72B0",edgecolor="white")
if np.isfinite(maj):
    ax1.axvline(maj,color="#888",lw=0.8,ls="--")
    ax1.text(maj+0.01,0.4,"majority-class\nbaseline (median)",fontsize=7,color="#555",va="bottom")
ax1.set_yticks(range(len(f))); ax1.set_yticklabels([f"{s} (n={n})" for s,n in zip(sites,ne)],fontsize=7.5)
ax1.set_xlabel("External macro-F1 (held-out site)"); ax1.set_xlim(0,1.02)
ax1.set_title("A  Per-site external performance",loc="left",fontweight="bold",fontsize=10.5)
for sp in ["top","right"]: ax1.spines[sp].set_visible(False)
# Panel B: detector-vs-F1 Spearman with CI
labels=[]; rhos=[]; los=[]; his=[]; cols=[]
for col,s,lab in orient:
    xo=(R[col]*s).values
    labels.append(lab)
    if np.std(xo)<1e-12:
        rhos.append(0.0); los.append(np.nan); his.append(np.nan); cols.append("#CCCCCC")
    else:
        rhos.append(spearmanr(xo,y).correlation); lo,hi=boot(xo,y); los.append(lo); his.append(hi)
        cols.append("#C44E52" if "TRACE" in lab else "#55A868")
yy=np.arange(len(labels))[::-1]
for i,(r,lo,hi,c) in enumerate(zip(rhos,los,his,cols)):
    yi=yy[i]
    if np.isnan(lo):
        ax2.scatter([0],[yi],marker="x",color="#999",s=28,zorder=3)
        ax2.text(0.03,yi,"constant (n/a)",fontsize=6.8,color="#999",va="center")
    else:
        ax2.plot([lo,hi],[yi,yi],color=c,lw=2.4,alpha=0.85,zorder=2)
        ax2.scatter([r],[yi],color=c,s=34,zorder=3,edgecolor="white",linewidth=0.6)
ax2.axvline(0,color="#444",lw=0.9)
ax2.set_yticks(yy); ax2.set_yticklabels(labels,fontsize=7.5)
ax2.set_xlim(-1,1); ax2.set_xlabel("Spearman $\\rho$ with external macro-F1  (95% bootstrap CI)")
ax2.set_title(f"B  Predictors of transportability ({len(R)} sites)",loc="left",fontweight="bold",fontsize=10.5)
for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig("figures/fig_realdata.pdf",bbox_inches="tight")
plt.savefig("figures/fig_realdata.png",dpi=300,bbox_inches="tight")
print("figure written. point estimates:")
for lab,r in zip(labels,rhos): print(f"  {lab:34s} rho={r:+.2f}")
