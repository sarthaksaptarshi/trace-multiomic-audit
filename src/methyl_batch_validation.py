#!/usr/bin/env python3
"""
TRACE methylation plate-batch study (run on YOUR laptop). Tests the novel claim:
does provenance/batch-quality auditing predict cross-BATCH transportability failure
better than predictive confidence, when a REAL methylation plate-shortcut exists?

Public data only:
 * methylation beta matrix (Illumina 450K or gene-level) from UCSC Xena
 * a REAL batch label (plate) -- from the TCGA barcode plate field or the MD Anderson
   TCGA Batch Effects tables -- supplied as a CSV: columns sample,plate
 * a non-methylation-derived subtype label (e.g. PAM50 from expression, or IHC ER/PR/HER2)

Leave-one-plate-out: train on all but one plate, test on the held-out plate; measure
external macro-F1 and AUROC; compute provenance (within-subtype across-TRAINING-plate
stability -- now a genuine batch provenance), predictive confidence, entropy; and
correlate each with realised external performance across plates (bootstrap CIs via
src/realdata_stats.py).
DISCLOSURE: the --maxprobes variance filter is computed on all samples, including
held-out plates (label-free, transductive). Report it in the methods.

Logic-tested on synthetic data; validate on your data. NOT a clinical study.

Usage:
  python methyl_batch_validation.py --beta BETA.tsv.gz --pheno CLIN.tsv \
    --subtype-col PAM50 --classes "LumA;Basal" --batch plate_map.csv \
    --topk 2000 --maxprobes 20000 --min-plate 15 --out methyl_results.csv
"""
import argparse, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score, roc_auc_score

def is_barcode(s): return isinstance(s,str) and s.upper().startswith("TCGA-")
def plate_from_barcode(bc):
    p=str(bc).split("-"); return p[5] if len(p)>=6 else None   # plate = 6th field of aliquot barcode
def present_class_auroc(y,p):
    """Macro one-vs-rest AUROC over the classes PRESENT in y (sklearn's multi-class OvR
    raises when a class is absent from a small held-out plate). Binary -> usual AUROC."""
    cs=np.unique(y)
    if len(cs)<2: return np.nan
    if p.shape[1]==2: return roc_auc_score(y,p[:,1])
    return float(np.mean([roc_auc_score((y==c).astype(int),p[:,c]) for c in cs]))
def topk(Z,y,k):
    F=f_classif(Z,y)[0]; F=np.where(np.isfinite(F),F,-np.inf); return np.argsort(F)[-k:]

def load_beta(path,maxprobes):
    df=pd.read_csv(path,sep="\t",index_col=0)      # probes x samples
    if np.mean([is_barcode(c) for c in df.columns[:20]])>np.mean([is_barcode(str(i)) for i in df.index[:20]]):
        df=df.T
    df.index=[str(i) for i in df.index]
    df=df.apply(pd.to_numeric,errors="coerce").astype("float32")   # samples x probes
    df=df.loc[:,df.std(axis=0)>0]
    if df.shape[1]>maxprobes:                                      # keep most-variable probes (laptop memory)
        v=df.var(axis=0).sort_values(ascending=False); df=df[v.index[:maxprobes]]
    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--beta",required=True); ap.add_argument("--pheno",required=True)
    ap.add_argument("--subtype-col",required=True); ap.add_argument("--classes",required=True)
    ap.add_argument("--batch",help="CSV sample,plate (if omitted, plate is taken from the barcode)")
    ap.add_argument("--topk",type=int,default=2000); ap.add_argument("--maxprobes",type=int,default=20000)
    ap.add_argument("--min-plate",type=int,default=15)
    ap.add_argument("--group-min",type=int,default=0,help="pool granular plates into batches each >= this many samples (greedy, sorted by plate ID); leave-one-BATCH-out"); ap.add_argument("--audit",type=int,default=20)
    ap.add_argument("--out",default="methyl_results.csv"); ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args(); rng=np.random.default_rng(a.seed); cls=[c.strip() for c in a.classes.split(";")]

    print("[1/4] labels + batch ..."); ph=pd.read_csv(a.pheno,sep="\t",index_col=0,low_memory=False)
    ph.index=[str(i) for i in ph.index]
    if a.subtype_col not in ph.columns: raise SystemExit("subtype-col missing. cols: "+", ".join(map(str,ph.columns[:40])))
    y=ph[a.subtype_col].astype(str)
    if a.batch:
        bm=pd.read_csv(a.batch); bm.columns=[c.lower() for c in bm.columns]
        plate=dict(zip(bm["sample"].astype(str),bm["plate"].astype(str)))
    else: plate=None
    print("[2/4] methylation beta (can take a minute, large) ..."); X=load_beta(a.beta,a.maxprobes)
    pl=pd.Series({s:(plate.get(s) if plate else plate_from_barcode(s)) for s in X.index})
    keep=[s for s in X.index if s in y.index and str(y.get(s)).strip() in cls and pl.get(s) not in (None,"nan","None")]
    X=X.loc[keep]; y=y.loc[keep].map({c:i for i,c in enumerate(cls)}).values; pl=pl.loc[keep].astype(str)
    if a.group_min>0:
        vc=pl.value_counts()
        order=sorted(vc.index); p2b={}; bid=0; run=0
        for p in order:
            p2b[p]="B%02d"%bid; run+=vc[p]
            if run>=a.group_min: bid+=1; run=0
        unit=pl.map(p2b)
        uc=unit.value_counts(); goodu=uc[uc>=a.min_plate].index
        keep=unit.isin(goodu).values
        X=X.values[keep]; y=y[keep]; str_plate=pl[keep].values; unit=unit[keep].values
        good=list(goodu); held=unit
        print(f"      grouped {len(order)} plates -> {len(set(held))} batches (>= {a.group_min} samples each)")
    else:
        vc=pl.value_counts(); good=vc[vc>=a.min_plate].index; m=pl.isin(good).values
        X=X.values[m]; y=y[m]; str_plate=pl[m].values; held=pl[m].values; good=list(good)
    print(f"      samples={len(X)} probes={X.shape[1]} classes={cls} units={len(good)}")
    if len(good)<4: raise SystemExit("Need >=4 plates with enough samples; lower --min-plate or pick a larger cohort.")

    print("[3/4] leave-one-plate-out ..."); rows=[]
    for ho in good:
        tr=np.where(held!=ho)[0]; te=np.where(held==ho)[0]
        if len(np.unique(y[te]))<2 or len(np.unique(y[tr]))<len(cls): continue
        p=rng.permutation(len(tr)); cut=int(0.8*len(p)); Xtr,ytr=X[tr][p[:cut]],y[tr][p[:cut]]; str_pl=str_plate[tr][p[:cut]]
        q=rng.permutation(len(te)); na=min(a.audit,len(q)//2); Xev,yev=X[te][q[na:]],y[te][q[na:]]
        imp=SimpleImputer(strategy="mean").fit(Xtr); sc=StandardScaler().fit(imp.transform(Xtr)); Z=lambda M:sc.transform(imp.transform(M)); sel=topk(Z(Xtr),ytr,a.topk)
        clf=LogisticRegression(max_iter=300,C=0.3,class_weight="balanced").fit(Z(Xtr)[:,sel],ytr)
        pe=clf.predict_proba(Z(Xev)[:,sel]); f1=f1_score(yev,clf.predict(Z(Xev)[:,sel]),average="macro")
        au=present_class_auroc(yev,pe)
        counts=np.bincount(yev,minlength=len(cls))
        conf=float(pe.max(1).mean()); ent=float((-(pe*np.log(pe+1e-9)).sum(1)).mean())
        # REAL batch provenance: within-subtype variance ACROSS TRAINING PLATES / total (low => batch-driven feature)
        bvals=[]
        for j in sel[:300]:
            w=[]
            for c in range(len(cls)):
                xi=Z(Xtr)[ytr==c,j]; si=str_pl[ytr==c]
                if len(xi)>5 and len(np.unique(si))>1:
                    w.append(np.var([xi[si==s].mean() for s in np.unique(si)])/(np.var(xi)+1e-6))
            if w: bvals.append(1-np.mean(w))
        prov=float(np.clip(np.mean(bvals),0,1)) if bvals else np.nan
        rows.append(dict(plate=ho,n_eval=len(yev),n_classes_present=int((counts>0).sum()),
                         class_counts="/".join(map(str,counts)),
                         external_F1=f1,external_AUROC=au,confidence=conf,entropy=ent,provenance=prov))
        print(f"   plate {ho}: nE={len(yev):3d} extF1={f1:.3f} AUROC={au:.3f} conf={conf:.3f} prov={prov:.3f}")
    R=pd.DataFrame(rows); R.to_csv(a.out,index=False); print(f"\n[wrote] {a.out} ({len(R)} plates)")
    if len(R)>=5:
        print("\n[4/4] Does provenance beat confidence at predicting cross-plate failure? (Spearman, oriented)")
        for out in ["external_F1","external_AUROC"]:
            ok=R[out].notna()
            for col,s in [("provenance",1),("confidence",1),("entropy",-1)]:
                rho=spearmanr((R.loc[ok,col]*s),R.loc[ok,out]).correlation
                print(f"   vs {out:14s} {col:11s} rho={rho:+.2f}  (n={ok.sum()})")
        print(f"   Bootstrap CIs and head-to-head: python src/realdata_stats.py --glioma {a.out}")

if __name__=="__main__": main()
