# Running the TRACE real-data validation on your laptop (TCGA via UCSC Xena)

All results you produce are yours; nothing here is pre-filled. Treat the first
run as a pipeline test, then do the non-circular version (see Step 2).

## Step 1 — Environment (10 min)
```
# install Python 3.10+ first (python.org or miniconda), then:
python -m venv trace-env
# Windows:  trace-env\Scripts\activate     macOS/Linux:  source trace-env/bin/activate
pip install numpy pandas scipy scikit-learn
# unzip TRACE_code_release.zip; the script is src/xena_validation.py
```

## Step 2 — Get the data from UCSC Xena (no login, open data)
1. Open https://xenabrowser.net/datapages/ and choose a TCGA cohort
   (e.g. "GDC TCGA Breast Cancer (BRCA)").
2. From that cohort's page, click each dataset and use **Download**:
   - a **gene expression** matrix (e.g. "HTSeq - FPKM" / "RSEM" / "pancan normalized"),
   - optionally **copy number (gene-level, Gistic2)**,
   - the **phenotype / clinical** matrix (has subtype + clinical columns).
3. If files end in `.gz`, decompress (`gunzip file.gz`).
4. Open the phenotype file and note the **exact subtype column name**
   (e.g. `PAM50Call_RNAseq`, `Subtype`, or an IHC receptor column).

### Choose labels to AVOID circularity (important)
- If label = **PAM50** (it is expression-derived), use **non-expression features**
  (copy number and/or methylation) — pass `--cnv` and omit expression, or vice-versa.
- Or use a **clinical label not derived from omics** (e.g. ER/PR/HER2 receptor
  status for BRCA) with expression features. Either choice is non-circular.
- Using expression features AND a PAM50 label is circular — only for a first
  pipeline test, not for the real result.

## Step 3 — Run
```
python src/xena_validation.py \
  --expr EXPR.tsv --pheno PHENO.tsv --subtype-col "PAM50Call_RNAseq" \
  --topk 500 --min-site 15 --audit 20 --out results.csv
# add  --cnv CNV.tsv  to include copy number
# add  --site-col COLNAME  if the phenotype has an explicit tissue-source-site column
```
The script does leave-one-tissue-source-site-out, prints per-site external macro-F1
and each detector, writes `results.csv`, and prints Spearman correlations across sites.

## Step 4 — Share back (paste here)
- the printed summary (per-site lines + the Spearman block), or upload `results.csv`;
- the cohort, which features (expression / CNV / methylation), the subtype label,
  the number of sites, and samples per site.
Then I will add confidence intervals, interpret it honestly, make the figure, and
fold it into the manuscript (or tell you plainly if it does not hold up).

## Caveats to keep in mind
- Tissue-source site is NOT a sequencing centre or batch (GDC/Xena are harmonised),
  so this tests population + residual technical shift — weaker than an independent cohort.
- Few sites = low statistical power; results are exploratory.
- If sample IDs in the two files don't overlap (0 samples), check barcode format
  (e.g. `TCGA-A2-A04P-01` vs `-01A`); trim to the first 4 blocks if needed.
- Expression matrices can be large (hundreds of MB); ensure disk space.

## Troubleshooting
- "subtype-col not in pheno": the script prints available columns — copy the exact name.
- "Need >=3 sites": lower `--min-site`, or pick a larger cohort.
- Slow: lower `--topk` (e.g. 300); the script already selects top-k genes.
