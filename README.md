# TRACE — a build-quality audit for multi-omic cancer subtyping

Code and analysis for the manuscript *"TRACE: a build-quality audit for multi-omic
cancer subtyping, and the limits of composite trust scores for transportability."*

This repository reproduces every figure, table and number in the paper. Results come
from (1) an auditable **simulation study** and (2) **real-data validation on three
open-access cohorts** — TCGA and METABRIC breast-cancer gene expression, and TCGA
glioma DNA methylation across sequencing plates. **This is a methods study, not a
clinical validation** — nothing here establishes clinical readiness, and the
simulated numbers must not be quoted as clinical performance.

---

## What the paper finds (in one paragraph)

A composite "trust score" built from four source-side build-quality diagnostics
(leakage, provenance, selection stability, calibration drift) appears to forecast
external performance in a hazard-level simulation, but that simulation is **circular**
— it generates performance from the same hazards the diagnostics observe. In a
**non-circular patient-level** evaluation the composite is only weakly predictive
(failure-detection AUROC ≈ 0.63–0.68) and is outperformed by predictive
confidence/entropy (≈ 0.91–0.93) and by direct estimation from a small labelled
audit (≈ 0.93). We tested whether **combining** build-quality with uncertainty helps
and found it does not beat the single strongest baseline: under realistic shift the
combined audit (0.94) does not improve on predictive confidence alone (0.93; paired
Δ = +0.014, 95% CI [−0.001, +0.034]; it does edge out the weaker 4-signal uncertainty
combination, Δ = +0.033 [+0.011, +0.065]), and the apparent advantage in a constructed
confident-error regime is a **scoring-direction artifact** — a direction-aware
confidence detector matches it (paired Δ = +0.001 [−0.006, +0.009]). In the **real-data validation**, a class-balanced classifier transports across a
genuine platform shift (TCGA↔METABRIC, AUROC 0.76–0.86), but across 21 pooled sites
**no detector's correlation with transport is distinguishable from zero**, raw or
adjusted for the cohort gap.
A third cohort (TCGA glioma DNA methylation across plates, a second omic type with
genuine cross-batch failures) shows the same direction: uncertainty weakly predicts
transport, provenance does not. **The consistent finding is negative/cautionary:
composite build-quality trust scores do not forecast transportability and do not
beat simple predictive confidence, across three real cohorts and two omic types.**

---

## Repository contents

```
src/
  patient_level.py        Non-circular patient-level validation (principal simulation result)
  comprehensive.py        Benchmark table: baselines, AUPRC, CIs, sensitivity
  combined_variant.py     Combined-audit experiments (incl. direction-aware confidence check)
  stress_test.py          Hazard-level / adversarial simulation (circular; labelled explanatory)
  prov_sweep.py           Provenance-misspecification sweep
  extras.py               Deferral logic, label-free calibration surrogate, counterexample
  xena_validation.py      Single-cohort TCGA real-data audit (leave-one-site-out, class-balanced)
  twocohort_validation.py Two-cohort TCGA + METABRIC validation (cross-cohort transfer + pooled sites)
  methyl_batch_validation.py  DNA-methylation audit across sequencing-plate batches (glioma cohort)
  realdata_stats.py       Bootstrap CIs, cohort-adjusted partial correlations, head-to-head vs confidence
  make_cohort_figs.py     Paper Figs. 3 and 4 (two-cohort, glioma) from the CSVs + realdata_stats.json
  make_genemap.py         Helper: map TCGA Ensembl IDs -> gene symbols (cross-cohort matching)
  tcga_validation.py      Legacy GDC-API scaffold (optional; Xena route preferred)
  results_io.py           Analysis scripts save results/*.json; figure scripts read them
  make_*_fig.py           Figure-generation scripts (read results/*.json; no hard-coded results)
  make_figures.py         Framework diagrams + SCHEMATIC figures (see "Figure provenance")
real_data/
  results_TCGA_BRCA.csv / results_TCGA_BRCA_fixed.csv   Single-cohort TCGA results
  results_twocohort.csv / results_twocohort_crosscohort.csv   Two-cohort results (21 sites + transfer)
  glioma_pilot4.csv       Glioma methylation, 4-class, leave-one-plate-batch-out
  make_realdata_fig.py    Real-data figure script (single-cohort CSV)
REAL_DATA_STEPS.md        How to download the data and run the real-data audits
requirements.txt          Python dependencies
CITATION.cff / .zenodo.json   Citation metadata   |   LICENSE  MIT
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate.bat
pip install -r requirements.txt   # numpy, pandas, scipy, scikit-learn, matplotlib  (Python >= 3.10)
```

The committed `results/` and `real_data/` outputs were produced with numpy 2.5.4, scipy 1.18.1,
scikit-learn 1.9.1, pandas 3.1.1 and matplotlib 3.11.2. With these versions every simulation
script reproduces its `results/*.json` byte-for-byte; other versions can differ in the last digit.

---

## Reproducing the simulation results

The simulated cohort (OMICS-SIM) is generated in code from fixed seeds — self-contained.

```bash
python src/patient_level.py      # non-circular validation
python src/comprehensive.py      # benchmark table (Table 1): baselines, AUPRC, CIs
python src/combined_variant.py   # combined-audit + direction-aware confidence check
python src/prov_sweep.py         # provenance-misspecification sweep
python src/stress_test.py        # hazard-level simulation (explanatory; circular by construction)
python src/extras.py             # deferral, leakage probe, audit size, surrogate counterexample
```

Run from the repository root. Each script writes `results/<name>.json`; then

```bash
python src/make_noncirc_fig.py; python src/make_combined_fig.py
python src/make_robust_fig.py;  python src/make_labelfree_fig.py
```

builds the results figures in `figures/` from those files. Point estimates are computed on
the full evaluation split; 95% CIs are percentile bootstraps. Failed deployment draws are
redrawn and the count is printed (`[disclosure]` line).

### Figure provenance

`src/make_figures.py` also draws figures that are **not** produced by any analysis here:
figs 5, 6, 9 and 11 are schematics drawn from hand-chosen parameters or formulas (stamped
"Schematic – illustrative values"), and figs 3, 4, 7, 10 and 12 plot hard-coded values
with no generating code in this repository. Do not present these as results.

---

## Reproducing the real-data validation

Data are **not** included (large and governed by their own terms). Download as in
`REAL_DATA_STEPS.md`. Sources: TCGA & TCGA-glioma via UCSC Xena
(https://xenabrowser.net/datapages/), METABRIC via cBioPortal
(https://www.cbioportal.org/study/summary?id=brca_metabric), and the real
batch/plate labels via the GDC biospecimen/aliquot barcodes.

**Single cohort (TCGA breast, leave-one-site-out):**
```bash
python src/xena_validation.py --expr TCGA-BRCA.star_fpkm-uq.tsv.gz --pheno TCGA-BRCA.clinical.tsv.gz \
  --subtype-col "primary_diagnosis.diagnoses" \
  --classes "Infiltrating duct carcinoma, NOS;Lobular carcinoma, NOS" \
  --balanced --topk 500 --min-site 15 --audit 20 --out results.csv
```

**Two cohorts (TCGA + METABRIC):**
```bash
python src/make_genemap.py --expr TCGA-BRCA.star_fpkm-uq.tsv.gz --out genemap.tsv   # needs: pip install mygene
python src/twocohort_validation.py --expr1 TCGA-BRCA.star_fpkm-uq.tsv.gz --pheno1 TCGA-BRCA.clinical.tsv.gz \
  --subtype-col1 "primary_diagnosis.diagnoses" --classes1 "Infiltrating duct carcinoma, NOS;Lobular carcinoma, NOS" \
  --expr2 data_mrna_illumina_microarray.txt --pheno2 data_clinical_patient.txt \
  --subtype-col2 "HISTOLOGICAL_SUBTYPE" --classes2 "Ductal/NST;Lobular" --site-col2 "COHORT" \
  --harmonize zscore --gene-map1 genemap.tsv --topk 500 --min-site 15 --audit 20 --out results_twocohort.csv
```

**Glioma DNA methylation (cross-plate batches):**
```bash
python src/methyl_batch_validation.py --beta TCGA.GBMLGG.sampleMap_HumanMethylation450.gz \
  --pheno glioma_label4.tsv --subtype-col label \
  --classes "GBM;Astrocytoma;Oligodendroglioma;Oligoastrocytoma" --batch plate_map_glioma.csv \
  --topk 2000 --maxprobes 20000 --min-plate 15 --group-min 25 --out glioma_pilot4.csv
```
**Statistics reported in the paper** (bootstrap CIs, cohort-adjusted partial Spearman,
head-to-head vs confidence):
```bash
python src/realdata_stats.py --single real_data/results_TCGA_BRCA_fixed.csv
python src/realdata_stats.py --twocohort real_data/results_twocohort.csv
python src/realdata_stats.py --glioma real_data/glioma_pilot4.csv
```
Run all three together to also write `results/realdata_stats.json`, then build the paper's
Fig. 3 (two cohorts) and Fig. 4 (glioma) from it:
```bash
python src/realdata_stats.py --single real_data/results_TCGA_BRCA_fixed.csv \
  --twocohort real_data/results_twocohort.csv --glioma real_data/glioma_pilot4.csv
python src/make_cohort_figs.py     # figures/fig_twocohort.*, figures/fig_glioma.*
```

The `real_data/` CSVs are the outputs of these commands after these fixes: (1) TCGA samples
are restricted to primary tumours, one per patient, in all three analyses (an earlier
two-cohort run included normal-tissue samples carrying the patient's carcinoma label, which
inflated the raw detector correlations; the glioma run included recurrent and normal samples
and let one patient appear in both a training and the held-out batch); (2) the glioma AUROC is
computed over the classes present on each plate; (3) the real-data provenance probe averages
over the top-ranked selected features (an earlier version used the lowest-ranked ones). With
duplicate patients removed, the glioma analysis has 12 evaluable plate-batches (was 14).
Glioblastoma is nearly absent from the evaluable held-out batches, so in practice that task is
close to three-class at evaluation.

---

## Honest scope and limitations

- Simulation results are illustrative; they are **not** clinical performance.
- The real-data analyses span three cohorts and two omic types but remain
  **illustrations, not a definitive validation**: tasks are binary/4-class with a
  logistic baseline, detector–performance correlations are imprecise with wide
  confidence intervals, and part of the raw signal is a between-cohort effect
  (reported via partial correlations).
- Per-cohort gene normalisation (two-cohort) and the methylation most-variable-probe filter
  use all samples of a cohort, including held-out sites/plates. This is label-free but
  transductive, and should be stated in the methods.
- Correlations across 14–21 sites/plates are exploratory; no multiplicity correction.
- Tissue-source site is an administrative label, not a sequencing batch; the glioma
  analysis uses the real sequencing **plate** as the batch variable.
- We adopt relevant principles of TRIPOD+AI as good practice but **do not claim
  conformance**; DECIDE-AI (live clinical evaluation) is out of scope.

---

## License and citation

MIT License (see `LICENSE`). If you use this code, please cite the paper and the
archived release (see `CITATION.cff`).

**Authors:** Sarthak Saptarshi Bandopadhyay and Manash Chanda, Meghnad Saha Institute
of Technology, Kolkata 700150, India (corresponding: sarthaksb@ieee.org; ORCID
0009-0002-4140-8144).

**Repository:** https://github.com/sarthaksaptarshi/trace-multiomic-audit

**DOI:** [10.5281/zenodo.23873596](https://doi.org/10.5281/zenodo.23873596) (Zenodo,
concept DOI — resolves to the latest version).
