#!/usr/bin/env python3
"""
TRACE two-cohort validation: TCGA (RNA-seq) + METABRIC (microarray), run on YOUR laptop.

Adds a genuine external cohort with a real platform + population shift, and pools
tissue-source sites across both so the detector-vs-transportability test is better
powered. Reads LOCAL files only; nothing is bundled. Logic-tested on synthetic
two-platform data. Builds on the bug-fixed single-cohort pipeline (NaN-safe gene
selection, constant genes dropped, class-balanced classifier).

Cross-platform handling: gene IDs are matched after stripping Ensembl version
suffixes and upper-casing symbols (optionally via --gene-map1/--gene-map2); each
cohort is then put on a common footing by per-gene normalisation WITHIN that cohort
(z-score or rank), which removes the platform offset/scale without using labels.
DISCLOSURE: this normalisation uses all samples of a cohort, including the samples of
a held-out site/cohort (label-free, transductive). Report it in the methods.

TCGA samples are restricted to primary tumours (sample codes 01/03), one per patient;
normal-tissue samples otherwise inherit the patient's histology label.

Usage (names/columns confirmed in step 2):
  python twocohort_validation.py \
    --expr1 TCGA_expr.tsv  --pheno1 TCGA_clin.tsv  --subtype-col1 "primary_diagnosis.diagnoses" \
    --classes1 "Infiltrating duct carcinoma, NOS;Lobular carcinoma, NOS" \
    --expr2 METABRIC_expr.txt --pheno2 METABRIC_clin.txt --subtype-col2 "HISTOLOGICAL_SUBTYPE" \
    --classes2 "Ductal/NST;Lobular" --site-col2 "COHORT" \
    --harmonize zscore --topk 500 --min-site 15 --audit 20 --out results_twocohort.csv
"""
import argparse, warnings, re, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

META_COLS={"entrez_gene_id","hugo_symbol","gene_symbol","gene","entrez","genes"}
MISS={"","nan","none","null","na","n/a","unknown","indeterminate","equivocal","not performed",
      "not available","not evaluated","not applicable","discrepancy","[not available]",
      "[not evaluated]","[unknown]","[discrepancy]","[not applicable]","[pending]"}
def is_miss(v): return (v is None) or (str(v).strip().lower() in MISS)
def norm_gene(g):
    g=str(g).strip()
    if g.upper().startswith("ENSG"): g=g.split(".")[0]   # strip Ensembl version
    return g.upper()
def tss(bc):
    p=str(bc).split("-"); return p[1] if (str(bc).upper().startswith("TCGA-") and len(p)>1) else None
def sample_code(bc):
    """TCGA sample-type code ('01' primary tumour, '11' solid normal, ...); None if not a TCGA barcode."""
    p=str(bc).split("-")
    return p[3][:2] if (str(bc).upper().startswith("TCGA-") and len(p)>3) else None
def tumour_only(ids):
    """Keep primary tumours (01/03) and one sample per TCGA patient; non-TCGA IDs pass through.
    Normal-tissue (11) and metastatic (06) samples inherit the patient's histology in the
    clinical matrix, so they must be dropped or they enter the analysis mislabelled."""
    out=[]; seen=set()
    for s in sorted(ids):
        sc=sample_code(s)
        if sc is not None and sc not in ("01","03"): continue
        pat="-".join(str(s).split("-")[:3]) if sc is not None else s
        if pat in seen: continue
        seen.add(pat); out.append(s)
    return out
def read_table(path):
    # skip leading comment lines (cBioPortal clinical files start with '#')
    sk=0
    with open(path,"r",encoding="utf-8",errors="ignore") as fh:
        for ln in fh:
            if ln.startswith("#"): sk+=1
            else: break
    return pd.read_csv(path,sep="\t",skiprows=sk,low_memory=False)

def load_expr(path,gene_map=None):
    """Any genes-in-rows matrix -> samples x genes, gene IDs normalised; drops metadata cols."""
    df=read_table(path); df=df.set_index(df.columns[0])
    df=df.drop(columns=[c for c in df.columns if str(c).strip().lower() in META_COLS],errors="ignore")
    df=df.T  # genes(rows)xsamples(cols) -> samples x genes
    genes=[norm_gene(g) for g in df.columns]
    if gene_map is not None: genes=[gene_map.get(g,g) for g in genes]
    df.columns=genes
    df=df.loc[:,~pd.Index(df.columns).duplicated()]           # drop dup genes
    df.index=[str(i) for i in df.index]
    return df.apply(pd.to_numeric,errors="coerce").astype("float32")

def load_pheno(path,id_col=None):
    df=read_table(path)
    if id_col and id_col in df.columns: df=df.set_index(id_col)
    elif "SAMPLE_ID" in df.columns: df=df.set_index("SAMPLE_ID")
    else: df=df.set_index(df.columns[0])
    df.index=[str(i) for i in df.index]
    return df

def harmonize(df,method):
    if method=="rank": return df.rank(axis=0,pct=True).astype("float32")
    return ((df-df.mean(0))/(df.std(0)+1e-9)).astype("float32")      # per-gene z within cohort

def topk_feats(Z,y,k):
    F=f_classif(Z,y)[0]; F=np.where(np.isfinite(F),F,-np.inf); return np.argsort(F)[-k:]
def ece(p,y,M=10):
    c=p.max(1); pr=p.argmax(1); a=(pr==y); b=np.linspace(0,1,M+1); e=0
    for i in range(M):
        m=(c>b[i])&(c<=b[i+1])
        if m.sum(): e+=m.mean()*abs(a[m].mean()-c[m].mean())
    return e
def ent(p): return float((-(p*np.log(p+1e-9)).sum(1)).mean())

def build_cohort(expr,pheno,subcol,classes,name,site_col,id_col,gene_map):
    X=load_expr(expr,gene_map); ph=load_pheno(pheno,id_col)
    if subcol not in ph.columns:
        print(f"\n!! --subtype-col for {name} not found. Columns:\n   "+"\n   ".join(map(str,ph.columns))); raise SystemExit
    y=ph[subcol].astype(str)
    cls=[c.strip() for c in classes.split(";")]
    keep=[s for s in X.index if s in y.index and str(y.get(s)).strip() in cls]
    n0=len(keep); keep=tumour_only(keep)
    if n0!=len(keep): print(f"      {name}: dropped {n0-len(keep)} non-primary-tumour / duplicate-patient samples")
    X=X.loc[keep]; y=y.loc[keep].map({c:i for i,c in enumerate(cls)})
    if site_col and site_col in ph.columns:
        site=ph[site_col].astype(str).loc[keep].map(lambda v:f"{name}:{v}")
    elif name.upper().startswith("TCGA") or any(str(s).upper().startswith("TCGA-") for s in keep[:5]):
        site=pd.Series({s:f"{name}:{tss(s)}" for s in keep})
    else:
        site=pd.Series({s:f"{name}:all" for s in keep})
    return X,y,site

def diagnostics(Xtr,ytr,Xsv,ysv,Xau,yau,Xev,yev,src_site,classes,topk,rng,cw):
    sc=StandardScaler().fit(Xtr); Z=lambda M:sc.transform(M)
    sel=topk_feats(Z(Xtr),ytr,topk)
    clf=LogisticRegression(max_iter=300,C=0.3,class_weight=cw).fit(Z(Xtr)[:,sel],ytr)
    f1=f1_score(yev,clf.predict(Z(Xev)[:,sel]),average="macro")
    pev=clf.predict_proba(Z(Xev)[:,sel]); pau=clf.predict_proba(Z(Xau)[:,sel]); psv=clf.predict_proba(Z(Xsv)[:,sel])
    try: auroc=roc_auc_score(yev,pev[:,1]) if len(classes)==2 else roc_auc_score(yev,pev,multi_class="ovr",average="macro")
    except Exception: auroc=np.nan
    conf=float(pev.max(1).mean()); entropy=ent(pev)
    def sb(): return np.concatenate([rng.choice(np.where(ytr==c)[0],(ytr==c).sum(),replace=True) for c in range(len(classes)) if (ytr==c).sum()>0])
    ens=float(np.stack([LogisticRegression(max_iter=200,C=0.3,class_weight=cw).fit(Z(Xtr)[(i:=sb())][:,sel],ytr[i]).predict_proba(Z(Xev)[:,sel]) for _ in range(5)]).var(0).mean())
    Xd=np.vstack([Z(Xtr)[:,sel],Z(Xev)[:,sel]]); yd=np.r_[np.zeros(len(Xtr)),np.ones(len(Xev))]
    dom=float(np.mean([roc_auc_score(yd[te],LogisticRegression(max_iter=200).fit(Xd[tr],yd[tr]).predict_proba(Xd[te])[:,1]) for tr,te in StratifiedKFold(3,shuffle=True,random_state=0).split(Xd,yd)]))
    sa=set(topk_feats(Z(Xtr)[:len(Xtr)//2],ytr[:len(Xtr)//2],topk)); sb2=set(topk_feats(Z(Xtr)[len(Xtr)//2:],ytr[len(Xtr)//2:],topk))
    stab=len(sa&sb2)/len(sa|sb2)
    bvals=[]
    for j in sel[-200:]:                      # sel is ascending by F: take the 200 top-ranked genes
        within=[]
        for c in range(len(classes)):
            xi=Z(Xtr)[ytr==c,j]; si=src_site[ytr==c]
            if len(xi)>5 and len(np.unique(si))>1: within.append(np.var([xi[si==s].mean() for s in np.unique(si)])/(np.var(xi)+1e-6))
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
    return dict(external_F1=f1,external_AUROC=auroc,confidence=conf,entropy=entropy,ensemble=ens,
                domain=dom,stability=stab,provenance=prov,leakage=leak,direct=direct,dECE=dECE)

def main():
    ap=argparse.ArgumentParser()
    for k in ["expr1","pheno1","subtype-col1","classes1","expr2","pheno2","subtype-col2","classes2"]:
        ap.add_argument("--"+k,required=True)
    ap.add_argument("--site-col1",default=None); ap.add_argument("--site-col2",default=None)
    ap.add_argument("--id-col1",default=None);   ap.add_argument("--id-col2",default=None)
    ap.add_argument("--gene-map1",default=None); ap.add_argument("--gene-map2",default=None)
    ap.add_argument("--name1",default="TCGA");   ap.add_argument("--name2",default="METABRIC")
    ap.add_argument("--harmonize",choices=["zscore","rank"],default="zscore")
    ap.add_argument("--topk",type=int,default=500); ap.add_argument("--min-site",type=int,default=15)
    ap.add_argument("--audit",type=int,default=20); ap.add_argument("--out",default="results_twocohort.csv")
    ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args(); rng=np.random.default_rng(a.seed); cw="balanced"

    def gmap(p):
        if not p: return None
        m=read_table(p); return {norm_gene(r.iloc[0]):str(r.iloc[1]).upper() for _,r in m.iterrows()}
    print("[1/4] loading cohort 1 ...")
    X1,y1,s1=build_cohort(a.expr1,a.pheno1,getattr(a,"subtype_col1"),a.classes1,a.name1,a.site_col1,a.id_col1,gmap(a.gene_map1))
    print("[2/4] loading cohort 2 ...")
    X2,y2,s2=build_cohort(a.expr2,a.pheno2,getattr(a,"subtype_col2"),a.classes2,a.name2,a.site_col2,a.id_col2,gmap(a.gene_map2))
    classesA=[c.strip() for c in a.classes1.split(";")]
    # shared genes
    shared=[g for g in X1.columns if g in set(X2.columns)]
    print(f"      shared genes: {len(shared)}  ({a.name1}:{X1.shape[1]}  {a.name2}:{X2.shape[1]})")
    if len(shared)<200: raise SystemExit("Too few shared genes -- gene IDs likely mismatched (e.g. Ensembl vs symbol). Use --gene-map1/--gene-map2 or a symbol-based matrix.")
    X1=X1[shared]; X2=X2[shared]
    # drop constant genes in either cohort, then per-cohort harmonise
    nz=(X1.std(0)>0)&(X2.std(0)>0); shared=[g for g,k in zip(shared,nz) if k]; X1=X1[shared]; X2=X2[shared]
    X1=harmonize(X1,a.harmonize); X2=harmonize(X2,a.harmonize)
    # pool
    X=pd.concat([X1,X2]); y=pd.concat([y1,y2]).astype(int); site=pd.concat([s1,s2]).astype(str)
    cohort=pd.Series(([a.name1]*len(X1))+([a.name2]*len(X2)),index=X.index)
    X=X.dropna(axis=1,how="any"); print(f"      pooled: n={len(X)} genes={X.shape[1]} classes={classesA}")

    print("[3/4] cross-cohort transfer (headline external validation) ...")
    def transfer(Xtr,ytr,Xte,yte):
        sc=StandardScaler().fit(Xtr.values); Z=lambda M:sc.transform(M)
        sel=topk_feats(Z(Xtr.values),ytr.values,a.topk)
        clf=LogisticRegression(max_iter=300,C=0.3,class_weight=cw).fit(Z(Xtr.values)[:,sel],ytr.values)
        p=clf.predict_proba(Z(Xte.values)[:,sel])
        au=roc_auc_score(yte.values,p[:,1]) if len(classesA)==2 else roc_auc_score(yte.values,p,multi_class="ovr",average="macro")
        f1=f1_score(yte.values,clf.predict(Z(Xte.values)[:,sel]),average="macro")
        return au,f1
    m1=cohort==a.name1; m2=cohort==a.name2
    au_ab,f1_ab=transfer(X[m1],y[m1],X[m2],y[m2]); au_ba,f1_ba=transfer(X[m2],y[m2],X[m1],y[m1])
    print(f"      train {a.name1} -> test {a.name2}: AUROC={au_ab:.3f}  macroF1={f1_ab:.3f}")
    print(f"      train {a.name2} -> test {a.name1}: AUROC={au_ba:.3f}  macroF1={f1_ba:.3f}")

    print("[4/4] pooled leave-one-site-out ...")
    vc=site.value_counts(); good=vc[vc>=a.min_site].index
    keep=site.isin(good); X,y,site,cohort=X[keep],y[keep],site[keep],cohort[keep]
    rows=[]
    for ho in good:
        tr=np.where(site.values!=ho)[0]; te=np.where(site.values==ho)[0]
        if len(np.unique(y.values[te]))<2 or len(np.unique(y.values[tr]))<len(classesA): continue
        p=rng.permutation(len(tr)); cut=int(0.8*len(p))
        Xtr=X.values[tr][p[:cut]]; ytr=y.values[tr][p[:cut]]; ssite=site.values[tr][p[:cut]]
        Xsv=X.values[tr][p[cut:]]; ysv=y.values[tr][p[cut:]]
        q=rng.permutation(len(te)); na=min(a.audit,len(q)//2)
        Xau=X.values[te][q[:na]]; yau=y.values[te][q[:na]]; Xev=X.values[te][q[na:]]; yev=y.values[te][q[na:]]
        d=diagnostics(Xtr,ytr,Xsv,ysv,Xau,yau,Xev,yev,ssite,classesA,a.topk,rng,cw)
        d.update(site=ho,cohort=cohort.values[te][0],n_eval=len(yev),minority_frac=float(min(np.bincount(yev,minlength=len(classesA)))/len(yev)))
        rows.append(d); print(f"   {ho:14s} nE={len(yev):3d} extF1={d['external_F1']:.3f} AUROC={d['external_AUROC']:.3f} prov={d['provenance']:.3f} leak={d['leakage']:.3f}")
    R=pd.DataFrame(rows)
    cols=["site","cohort","n_eval","minority_frac","external_F1","external_AUROC","confidence","entropy","ensemble","domain","stability","provenance","leakage","direct","dECE"]
    R=R[[c for c in cols if c in R.columns]]; R.to_csv(a.out,index=False)
    # append cross-cohort headline row to CSV as a comment-free summary file
    pd.DataFrame([{"transfer":f"{a.name1}->{a.name2}","AUROC":au_ab,"macroF1":f1_ab},
                  {"transfer":f"{a.name2}->{a.name1}","AUROC":au_ba,"macroF1":f1_ba}]).to_csv(a.out.replace(".csv","_crosscohort.csv"),index=False)
    print(f"\n[wrote] {a.out}  ({len(R)} sites)  and  {a.out.replace('.csv','_crosscohort.csv')}")
    if len(R)>=5:
        print("\nSpearman vs external AUROC across pooled sites (oriented; +rho = better predictor):")
        for col,s in [("provenance",1),("leakage",-1),("ensemble",-1),("domain",-1),("stability",1),
                      ("confidence",1),("entropy",-1),("direct",1),("dECE",-1)]:
            xo=R[col]*s
            if xo.std()<1e-12: print(f"   {col:11s} constant"); continue
            print(f"   {col:11s} rho={spearmanr(xo,R['external_AUROC']).correlation:+.2f}")
        print(f"\nBootstrap CIs and cohort-adjusted partial correlations: python src/realdata_stats.py --twocohort {a.out}")

if __name__=="__main__": main()
