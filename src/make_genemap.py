#!/usr/bin/env python3
"""
Build an Ensembl -> gene-symbol map for a Xena TCGA expression file, so its genes
can be matched to a symbol-based cohort (e.g. METABRIC). Run once; pass the output
to twocohort_validation.py via --gene-map1.

Needs the small 'mygene' package (looks symbols up online):   pip install mygene

Usage:
  python src\\make_genemap.py --expr TCGA-BRCA.star_fpkm-uq.tsv.gz --out genemap.tsv
"""
import argparse, sys, pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--expr",required=True); ap.add_argument("--out",default="genemap.tsv")
    a=ap.parse_args()
    print("reading gene IDs from",a.expr,"...")
    ids=pd.read_csv(a.expr,sep="\t",usecols=[0]).iloc[:,0].astype(str).tolist()
    base=sorted({i.split(".")[0] for i in ids if i.upper().startswith("ENSG")})
    print(f"{len(base)} Ensembl gene IDs to translate ...")
    if not base: sys.exit("No Ensembl IDs found in the first column; is this the right expression file?")
    try:
        import mygene
    except ImportError:
        sys.exit("Please run:  pip install mygene   then run this script again.")
    mg=mygene.MyGeneInfo()
    res=mg.querymany(base,scopes="ensembl.gene",fields="symbol",species="human",returnall=True)
    rows=[(d["query"],d["symbol"]) for d in res["out"] if d.get("symbol") and not d.get("notfound")]
    pd.DataFrame(rows,columns=["ensembl","symbol"]).to_csv(a.out,sep="\t",index=False)
    print(f"wrote {a.out} with {len(rows)} Ensembl->symbol mappings "
          f"({len(rows)*100//max(len(base),1)}% of genes mapped)")

if __name__=="__main__": main()
