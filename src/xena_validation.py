#!/usr/bin/env python3
"""
TRACE real-data validation on TCGA via UCSC Xena files (run on YOUR laptop).
Leave-one-tissue-source-site-out multi-omic subtyping; compares TRACE build-quality
diagnostics, uncertainty baselines, direct estimation, and a combined audit against
the REAL external macro-F1 measured on held-out sites.

NO data bundled; reads LOCAL files only. Logic-tested on synthetic Xena-shaped files.
Sanity-check every number.

CAVEATS to state in any write-up:
 * Tissue-source site is NOT a sequencing centre/batch (GDC/Xena are harmonised); this
   tests population + residual technical shift, weaker than an independent cohort.
 * CIRCULARITY: if the label is derived from the feature assay (e.g. PAM50 from
   expression), the test is circular. Use a non-expression label (e.g. ER/PR/HER2
   receptor status) with expression, or non-expression features with a PAM50 label.
 * Provenance here uses tissue-source site as a PSEUDO-batch (no true batch in Xena).

Usage:
  python xena_validation.py --expr EXPR.tsv --pheno PHENO.tsv --subtype-col COL \
      [--cnv CNV.tsv] [--classes "A;B"] [--topk 500] \
      [--min-site 15] [--audit 20] [--out results.csv]
"""
import argparse, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# TCGA placeholder / missing tokens dropped from labels
MISS={"", "nan","none","null","na","n/a","unknown","indeterminate","equivocal",
      "not performed","not available","not evaluated","not applicable","discrepancy",
      "[not available]","[not evaluated]","[unknown]","[discrepancy]","[not applicable]",
      "[pending]","[completed]","[uncertain]"}
def is_miss(v): return (v is None) or (str(v).strip().lower() in MISS)

def is_barcode(s): return isinstance(s,str) and s.upper().startswith("TCGA-")
def tss(bc): return bc.split("-")[1] if is_barcode(bc) else None
def sample_code(bc):
    p=bc.split("-"); return p[3][:2] if len(p)>3 else None

def load_matrix(path):
    """Xena genomic matrix -> samples x features (auto-orient, float32)."""
    df=pd.read_csv(path,sep="\t",index_col=0)
    fc=np.mean([is_barcode(c) for c in df.columns[:20]])
    fr=np.mean([is_barcode(str(i)) for i in df.index[:20]])
    if fc>fr: df=df.T
    df.index=[str(i) for i in df.index]
    try: df=df.astype("float32")
    except Exception: df=df.apply(pd.to_numeric,errors="coerce").astype("float32")
    return df

def ece(p,y,M=10):
    c=p.max(1); pr=p.argmax(1); a=(pr==y); b=np.linspace(0,1,M+1); e=0
    for i in range(M):
        m=(c>b[i])&(c<=b[i+1])
        if m.sum(): e+=m.mean()*abs(a[m].mean()-c[m].mean())
    return e
def ent(p): return float((-(p*np.log(p+1e-9)).sum(1)).mean())
def topk_feats(Zmat,yvec,k):
    """univariate F-selection, NaN/inf-safe (constant genes must not win the sort)."""
    F=f_classif(Zmat,yvec)[0]; F=np.where(np.isfinite(F),F,-np.inf)
    return np.argsort(F)[-k:]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--expr",required=True); ap.add_argument("--pheno",required=True)
    ap.add_argument("--subtype-col",required=True); ap.add_argument("--cnv")
    ap.add_argument("--classes",default=None,help='restrict to these labels, SEMICOLON-separated, e.g. "Infiltrating duct carcinoma, NOS;Lobular carcinoma, NOS"')
    ap.add_argument("--site-col",default=None)
    ap.add_argument("--topk",type=int,default=500); ap.add_argument("--min-site",type=int,default=15)
    ap.add_argument("--audit",type=int,default=20); ap.add_argument("--out",default="results.csv")
    ap.add_argument("--balanced",action="store_true",help="class_weight=balanced so the model actually learns the minority subtype (recommended)")
    ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args(); rng=np.random.default_rng(a.seed)
    cw="balanced" if a.balanced else None

    print("[1/4] loading phenotype ...")
    ph=pd.read_csv(a.pheno,sep="\t",index_col=0,low_memory=False); ph.index=[str(i) for i in ph.index]
    if a.subtype_col not in ph.columns:
        print("\n!! --subtype-col not found. Available columns:\n")
        for c in ph.columns: print("   ",c)
        raise SystemExit("\nPick one of the above and pass it to --subtype-col.")
    y=ph[a.subtype_col].astype(str)
    site = ph[a.site_col].astype(str) if (a.site_col and a.site_col in ph.columns) else pd.Series({s:tss(s) for s in ph.index})

    print("[2/4] loading expression (can take a minute) ...")
    X=load_matrix(a.expr)
    if a.cnv:
        C=load_matrix(a.cnv).add_prefix("cnv_"); X=X.join(C,how="inner")

    print("[3/4] aligning & filtering ...")
    keep=[s for s in X.index if s in y.index and not is_miss(y.get(s))
          and (sample_code(s) in (None,"01","03"))]
    seen=set(); dedup=[]                      # one sample per patient (TCGA-XX-XXXX)
    for s in sorted(keep):
        pat="-".join(s.split("-")[:3]) if is_barcode(s) else s
        if pat not in seen: seen.add(pat); dedup.append(s)
    if len(dedup)!=len(keep): print(f"      dropped {len(keep)-len(dedup)} duplicate-patient samples")
    keep=dedup
    X=X.loc[keep].dropna(axis=1,how="any"); y=y.loc[X.index]; site=site.loc[X.index]
    nz=X.std(axis=0)>0; X=X.loc[:,nz]  # drop constant (unexpressed) genes
    if a.classes:
        allowed=set(c.strip() for c in a.classes.split(";")); m=y.isin(allowed)
        X,y,site=X[m],y[m],site[m]
    vc=site.value_counts(); good=vc[vc>=a.min_site].index
    m=site.isin(good); X,y,site=X[m],y[m],site[m]
    classes=sorted(y.unique()); ymap={c:i for i,c in enumerate(classes)}; yv=y.map(ymap).values
    print(f"      samples={len(X)} features={X.shape[1]} classes={classes} sites={len(good)}")
    if len(classes)<2: raise SystemExit("Only one label class left after filtering; pick a different --subtype-col or --classes.")
    if len(good)<3: raise SystemExit("Need >=3 sites with enough samples; lower --min-site.")

    print("[4/4] leave-one-site-out ...")
    rows=[]
    for ho in good:
        tr_idx=np.where(~site.isin([ho]).values)[0]; te_idx=np.where(site.values==ho)[0]
        if len(np.unique(yv[te_idx]))<2 or len(np.unique(yv[tr_idx]))<len(classes): continue
        Xtr_all,ytr_all=X.values[tr_idx],yv[tr_idx]; Xte,yte=X.values[te_idx],yv[te_idx]
        p=rng.permutation(len(tr_idx)); cut=int(0.8*len(p))
        Xtr,ytr=Xtr_all[p[:cut]],ytr_all[p[:cut]]; Xsv,ysv=Xtr_all[p[cut:]],ytr_all[p[cut:]]
        q=rng.permutation(len(te_idx)); na=min(a.audit,len(q)//2)
        Xau,yau=Xte[q[:na]],yte[q[:na]]; Xev,yev=Xte[q[na:]],yte[q[na:]]
        sc=StandardScaler().fit(Xtr); Z=lambda M:sc.transform(M)
        sel=topk_feats(Z(Xtr),ytr,a.topk)
        clf=LogisticRegression(max_iter=300,C=0.3,class_weight=cw).fit(Z(Xtr)[:,sel],ytr)
        f1=f1_score(yev,clf.predict(Z(Xev)[:,sel]),average="macro")
        pev=clf.predict_proba(Z(Xev)[:,sel]); pau=clf.predict_proba(Z(Xau)[:,sel]); psv=clf.predict_proba(Z(Xsv)[:,sel])
        try:
            auroc=roc_auc_score(yev,pev[:,1]) if len(classes)==2 else roc_auc_score(yev,pev,multi_class="ovr",average="macro")
        except Exception: auroc=np.nan
        minfrac=float(min(np.bincount(yev,minlength=len(classes)))/len(yev))
        conf=float(pev.max(1).mean()); entropy=ent(pev)
        def sboot(): return np.concatenate([rng.choice(np.where(ytr==c)[0],(ytr==c).sum(),replace=True) for c in range(len(classes)) if (ytr==c).sum()>0])
        ens=float(np.stack([LogisticRegression(max_iter=200,C=0.3,class_weight=cw).fit(Z(Xtr)[(i:=sboot())][:,sel],ytr[i]).predict_proba(Z(Xev)[:,sel]) for _ in range(5)]).var(0).mean())
        Xd=np.vstack([Z(Xtr)[:,sel],Z(Xev)[:,sel]]); yd=np.r_[np.zeros(len(Xtr)),np.ones(len(Xev))]
        dom=float(np.mean([roc_auc_score(yd[te],LogisticRegression(max_iter=200).fit(Xd[tr],yd[tr]).predict_proba(Xd[te])[:,1]) for tr,te in StratifiedKFold(3,shuffle=True,random_state=0).split(Xd,yd)]))
        sa=set(topk_feats(Z(Xtr)[:len(Xtr)//2],ytr[:len(Xtr)//2],a.topk))
        sb=set(topk_feats(Z(Xtr)[len(Xtr)//2:],ytr[len(Xtr)//2:],a.topk))
        stab=len(sa&sb)/len(sa|sb)
        src_site=site.values[tr_idx][p[:cut]]; bvals=[]
        for j in sel[-200:]:                  # sel is ascending by F: take the 200 top-ranked genes
            within=[]
            for c in range(len(classes)):
                xi=Z(Xtr)[ytr==c,j]; si=src_site[ytr==c]
                if len(xi)>5 and len(np.unique(si))>1:
                    within.append(np.var([xi[si==s].mean() for s in np.unique(si)])/(np.var(xi)+1e-6))
            if within: bvals.append(1-np.mean(within))
        prov=float(np.clip(np.mean(bvals),0,1)) if bvals else np.nan
        yp=rng.permutation(ytr)
        def cvacc(outside):
            acc=[]
            for tri,tei in StratifiedKFold(3,shuffle=True,random_state=1).split(Z(Xtr),yp):
                fe=topk_feats(Z(Xtr),yp,25) if outside else topk_feats(Z(Xtr)[tri],yp[tri],25)
                acc.append((LogisticRegression(max_iter=150).fit(Z(Xtr)[tri][:,fe],yp[tri]).predict(Z(Xtr)[tei][:,fe])==yp[tei]).mean())
            return np.mean(acc)
        leak=float(max(0,cvacc(True)-cvacc(False)))
        direct=f1_score(yau,clf.predict(Z(Xau)[:,sel]),average="macro")
        dECE=float(ece(pau,yau)-ece(psv,ysv))
        rows.append(dict(site=ho,n_eval=len(yev),external_F1=f1,external_AUROC=auroc,minority_frac=minfrac,confidence=conf,entropy=entropy,
                         ensemble=ens,domain=dom,stability=stab,provenance=prov,leakage=leak,
                         direct=direct,dECE=dECE))
        print(f"   site {ho}: nE={len(yev):3d} min%={minfrac:.2f} extF1={f1:.3f} AUROC={auroc:.3f} conf={conf:.3f} prov={prov:.3f} direct={direct:.3f}")
    R=pd.DataFrame(rows); R.to_csv(a.out,index=False)
    print(f"\n[wrote] {a.out}  ({len(R)} sites)")
    if len(R)>=4:
        print("\nSpearman correlation with external macro-F1 across sites (higher=better predictor of real performance):")
        for col,sign in [("confidence",1),("entropy",-1),("ensemble",-1),("domain",-1),
                         ("stability",1),("provenance",1),("leakage",-1),("direct",1)]:
            rho=spearmanr(R[col]*sign, R["external_F1"]).correlation
            print(f"   {col:12s} rho={rho:+.2f}")
        def z(x): x=pd.to_numeric(x,errors='coerce'); return (x-x.mean())/(x.std()+1e-9)
        comb=z(R["confidence"])+z(R["stability"])+z(R["provenance"].fillna(R["provenance"].mean()))
        print(f"   combined(conf+stab+prov) rho={spearmanr(comb,R['external_F1']).correlation:+.2f}")
        if R["external_AUROC"].notna().sum()>=4 and R["external_AUROC"].std()>1e-9:
            print("\nSame, but vs external AUROC (more sensitive than macro-F1 when classes are imbalanced):")
            for col,sign in [("confidence",1),("entropy",-1),("ensemble",-1),("domain",-1),
                             ("stability",1),("provenance",1),("leakage",-1),("direct",1)]:
                xo=R[col]*sign
                if xo.std()<1e-12: print(f"   {col:12s} rho= n/a (constant)"); continue
                ok=R["external_AUROC"].notna()   # one-class sites have no AUROC
                print(f"   {col:12s} rho={spearmanr(xo[ok],R['external_AUROC'][ok]).correlation:+.2f}")
        print("\nNOTE: with few sites this is LOW-POWERED; report CIs and treat as exploratory.")
    else:
        print("Too few evaluable sites for correlation; per-site table only.")

if __name__=="__main__": main()
