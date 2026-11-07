#!/usr/bin/env python3
"""
Statistics for the real-data result CSVs (the numbers reported in the paper's
real-data section). Reads the per-site / per-plate CSVs written by
xena_validation.py, twocohort_validation.py and methyl_batch_validation.py.

For each detector (oriented so that +rho = detector tracks better transport):
  * Spearman rho vs the external outcome, with a percentile bootstrap 95% CI over
    sites/plates;
  * for the pooled two-cohort table, the cohort-adjusted PARTIAL Spearman (ranks of
    detector and outcome each residualised on a cohort indicator), bootstrapped
    within cohort so both cohorts are always present;
  * head-to-head: paired bootstrap of rho(detector) - rho(confidence).
No multiplicity correction is applied; with ~15-21 units these are exploratory.

Usage:
  python src/realdata_stats.py --single  real_data/results_TCGA_BRCA_fixed.csv
  python src/realdata_stats.py --twocohort real_data/results_twocohort.csv
  python src/realdata_stats.py --glioma  real_data/glioma_pilot4.csv
"""
import argparse, numpy as np, pandas as pd
from scipy.stats import rankdata, pearsonr

ORIENT={"confidence":1,"entropy":-1,"ensemble":-1,"domain":-1,"stability":1,
        "provenance":1,"leakage":-1,"direct":1,"dECE":-1}

def spear(x,y):
    if np.std(x)<1e-12 or np.std(y)<1e-12: return np.nan
    return pearsonr(rankdata(x),rankdata(y))[0]

def partial_spear(x,y,g):
    rx,ry=rankdata(x),rankdata(y)
    G=np.column_stack([np.ones(len(g)),(g==g[0]).astype(float)])
    res=lambda v: v-G@np.linalg.lstsq(G,v,rcond=None)[0]
    ex,ey=res(rx),res(ry)
    if np.std(ex)<1e-12 or np.std(ey)<1e-12: return np.nan
    return pearsonr(ex,ey)[0]

def boot_idx(n,groups,rng):
    if groups is None: return rng.integers(0,n,n)
    idx=[]
    for gv in np.unique(groups):
        w=np.where(groups==gv)[0]; idx.append(rng.choice(w,len(w),replace=True))
    return np.concatenate(idx)

def boot(stat,cols,groups,B,rng):
    n=len(cols[0]); pt=stat(*cols); out=[]
    for _ in range(B):
        i=boot_idx(n,groups,rng)
        v=stat(*[c[i] for c in cols])
        if np.isfinite(v): out.append(v)
    lo,hi=(np.percentile(out,2.5),np.percentile(out,97.5)) if out else (np.nan,np.nan)
    return pt,lo,hi

def report(R,outcome,B,rng,cohort_col=None,label=""):
    R=R[R[outcome].notna()].reset_index(drop=True)
    y=R[outcome].values.astype(float)
    g=R[cohort_col].values if (cohort_col and cohort_col in R.columns and R[cohort_col].nunique()>1) else None
    dets=[c for c in ORIENT if c in R.columns and R[c].notna().all()]
    print(f"\n== {label}  outcome={outcome}  n={len(R)} units ==")
    hdr="  %-11s %-24s"%("detector","Spearman rho [95% CI]")
    if g is not None: hdr+=" %-26s"%"partial | cohort [95% CI]"
    if "confidence" in dets: hdr+=" %-26s"%"rho - rho(conf) [95% CI]"
    print(hdr)
    for d in dets:
        x=R[d].values.astype(float)*ORIENT[d]
        line="  %-11s %+.2f [%+.2f,%+.2f]     "%(d,*boot(spear,[x,y],g,B,rng))
        if g is not None:
            line+=" %+.2f [%+.2f,%+.2f]       "%boot(lambda a,b,gg:partial_spear(a,b,gg),[x,y,g],g,B,rng)
        if "confidence" in dets and d!="confidence":
            c=R["confidence"].values.astype(float)
            line+=" %+.2f [%+.2f,%+.2f]"%boot(lambda a,b,cc:spear(a,b)-spear(cc,b),[x,y,c],g,B,rng)
        print(line)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--single"); ap.add_argument("--twocohort"); ap.add_argument("--glioma")
    ap.add_argument("--B",type=int,default=5000); ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args(); rng=np.random.default_rng(a.seed)
    if not (a.single or a.twocohort or a.glioma): ap.error("give at least one of --single/--twocohort/--glioma")
    if a.single:
        R=pd.read_csv(a.single)
        for out in ["external_F1","external_AUROC"]: report(R,out,a.B,rng,label="single cohort")
    if a.twocohort:
        R=pd.read_csv(a.twocohort)
        for out in ["external_AUROC","external_F1"]: report(R,out,a.B,rng,cohort_col="cohort",label="two-cohort pooled sites")
    if a.glioma:
        R=pd.read_csv(a.glioma)
        for out in ["external_F1","external_AUROC"]: report(R,out,a.B,rng,label="glioma plates")
    print("\nExploratory: no multiplicity correction; wide CIs are expected with this few units.")

if __name__=="__main__": main()
