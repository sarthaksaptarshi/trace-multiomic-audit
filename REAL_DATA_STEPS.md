# Reproducing the TRACE real-data analyses

All real-data analyses use **open-access** data only; nothing is bundled in this
repository (the matrices are large and governed by their sources' terms). Download
the inputs yourself as below, then run the scripts in `src/`. These are research
illustrations, **not** a clinical validation.

Environment: Python >= 3.10 with the packages in `requirements.txt`
(`pip install -r requirements.txt`). The scripts read `.gz` files directly.

---

## 1. TCGA breast cancer, single cohort (leave-one-site-out)

Data from UCSC Xena (https://xenabrowser.net/datapages/), GDC TCGA BRCA cohort:
gene expression (STAR FPKM-UQ) and the clinical/phenotype matrix. The label is the
pathology-assigned histology (`primary_diagnosis.diagnoses`), which is non-circular
with expression features.

```
python src/xena_validation.py --expr TCGA-BRCA.star_fpkm-uq.tsv.gz \
  --pheno TCGA-BRCA.clinical.tsv.gz --subtype-col "primary_diagnosis.diagnoses" \
  --classes "Infiltrating duct carcinoma, NOS;Lobular carcinoma, NOS" \
  --balanced --topk 500 --min-site 15 --audit 20 --out results.csv
```

## 2. TCGA + METABRIC, two cohorts (cross-cohort transfer + pooled sites)

Add METABRIC (microarray expression + clinical) from cBioPortal
(https://www.cbioportal.org/study/summary?id=brca_metabric). TCGA uses Ensembl gene
IDs and METABRIC uses symbols, so first build a mapping (needs `mygene`):

```
python src/make_genemap.py --expr TCGA-BRCA.star_fpkm-uq.tsv.gz --out genemap.tsv
python src/twocohort_validation.py \
  --expr1 TCGA-BRCA.star_fpkm-uq.tsv.gz --pheno1 TCGA-BRCA.clinical.tsv.gz \
  --subtype-col1 "primary_diagnosis.diagnoses" \
  --classes1 "Infiltrating duct carcinoma, NOS;Lobular carcinoma, NOS" \
  --expr2 data_mrna_illumina_microarray.txt --pheno2 data_clinical_patient.txt \
  --subtype-col2 "HISTOLOGICAL_SUBTYPE" --classes2 "Ductal/NST;Lobular" \
  --site-col2 "COHORT" --harmonize zscore --gene-map1 genemap.tsv \
  --topk 500 --min-site 15 --audit 20 --out results_twocohort.csv
```

## 3. TCGA glioma DNA methylation (cross-plate batches)

Data from UCSC Xena, TCGA GBMLGG cohort: Methylation450k beta matrix and the
clinical matrix (histology). Build a 4-class histology label from the clinical
matrix, and a real sequencing-**plate** batch map from the GDC aliquot barcodes
(the plate is the 6th barcode field); fetch the aliquot barcodes via the GDC files
API or the project Biospecimen download for TCGA-GBM and TCGA-LGG. Then:

```
python src/methyl_batch_validation.py \
  --beta TCGA.GBMLGG.sampleMap_HumanMethylation450.gz --pheno glioma_label4.tsv \
  --subtype-col label --classes "GBM;Astrocytoma;Oligodendroglioma;Oligoastrocytoma" \
  --batch plate_map_glioma.csv --topk 2000 --maxprobes 20000 \
  --min-plate 15 --group-min 25 --out glioma_pilot4.csv
```

The CSV outputs of these three analyses are in `real_data/`.

---

## Notes and caveats

- Use a non-circular label: pathology histology or IHC receptor status with omic
  features, never a label derived from the same assay used as features.
- The TCGA sample barcodes in Xena matrices are sample-level (`TCGA-XX-XXXX-01`);
  the genuine sequencing **plate** is only in the full aliquot barcode, obtained
  from the GDC (used in analysis 3).
- Methylation beta matrices contain missing values; `methyl_batch_validation.py`
  mean-imputes on the training split before standardising.
- TCGA samples are restricted to primary tumours (sample codes 01/03), one per patient.
  Normal-tissue samples otherwise inherit the patient's histology label.
- With few sites/batches these analyses are exploratory. Bootstrap confidence intervals
  and cohort-adjusted partial correlations come from `src/realdata_stats.py`.
- `glioma_label4.tsv` and `plate_map_glioma.csv` are built by hand as described in step 3;
  record the exact clinical columns and GDC query you used so others can rebuild them.
