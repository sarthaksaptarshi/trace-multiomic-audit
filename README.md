# TRACE — a build-quality audit for multi-omic cancer subtyping

Code and analysis for the manuscript *"TRACE: a build-quality audit for multi-omic
cancer subtyping, and the limits of composite trust scores for transportability."*

This repository reproduces every figure, table and number in the paper. Results come
from (1) an auditable **simulation study** and (2) a **two-cohort real-data
validation** on open-access TCGA and METABRIC breast-cancer data. **This is a
methods study, not a clinical validation** — nothing here establishes clinical
readiness, and the simulated numbers must not be quoted as clinical performance.

---

## What the paper finds (in one paragraph)

A composite "trust score" built from four source-side build-quality diagnostics
(leakage, provenance, selection stability, calibration drift) appears to forecast
external performance in a hazard-level simulation, but that simulation is **circular**
— it generates performance from the same hazards the diagnostics observe. In a
**non-circular patient-level** evaluation the composite is only weakly predictive
(failure-detection AUROC ≈ 0.63–0.68) and is beaten by predictive confidence/entropy
(≈ 0.91–0.93) and by direct estimation from a small labelled audit (≈ 0.93).
Build-quality diagnostics are nonetheless **complementary** to uncertainty in
specific regimes. In the **two-cohort real-data validation**, a class-balanced
classifier transports across a genuine platform shift (TCGA↔METABRIC,
AUROC 0.76–0.86), and across 21 pooled sites several build-quality/shift probes
predict cross-site AUROC with 95% intervals excluding zero — most robustly **leakage
susceptibility** (which survives controlling for the cohort gap), with calibration
drift close behind, while provenance's raw signal is a cohort artefact and
uncertainty stays weak. The honest recommendation is a **combined audit**, not any
single probe.

---

## Repository contents

```
src/
  patient_level.py        Non-circular patient-level validation (principal simulation result)
  stress_test.py          Hazard-level / adversarial simulation (circular; labelled as explanatory)
  comprehensive.py        Baselines, AUPRC, audit-size and sensitivity analyses
  prov_sweep.py           Provenance-misspecification sweep
  combined_variant.py     Build-quality × uncertainty complementarity experiments
  extras.py               Deferral logic, label-free calibration surrogate, counterexample
  xena_validation.py      Single-cohort TCGA real-data audit (leave-one-site-out, class-balanced)
  twocohort_validation.py Two-cohort TCGA + METABRIC validation (cross-cohort transfer + pooled sites)
  make_genemap.py         Helper: map TCGA Ensembl IDs -> gene symbols (for cross-cohort matching)
  tcga_validation.py      Legacy GDC-API scaffold (optional; Xena route is preferred)
  make_*_fig.py           Figure-generation scripts
real_data/
  results_TCGA_BRCA_fixed.csv        Single-cohort TCGA results (bug-fixed run)
  results_twocohort.csv              Two-cohort pooled per-site results (21 sites)
  results_twocohort_crosscohort.csv  Cross-cohort transfer AUROCs
  make_realdata_fig.py               Real-data figure script
REAL_DATA_STEPS.md        Step-by-step guide to downloading the data and running the real-data audits
requirements.txt          Python dependencies
CITATION.cff / .zenodo.json   Citation metadata
LICENSE                   MIT
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate.bat
pip install -r requirements.txt
```
Python ≥ 3.10. Core dependencies: numpy, pandas, scipy, scikit-learn, matplotlib.

---

## Reproducing the simulation results

The simulated cohort (OMICS-SIM) is generated in code from fixed random seeds, so the
scripts are self-contained — no external data needed.

```bash
python src/patient_level.py      # non-circular validation (principal result + Fig.)
python src/comprehensive.py      # baselines, AUPRC, sensitivity
python src/combined_variant.py   # complementarity experiments
python src/prov_sweep.py         # provenance-misspecification sweep
python src/stress_test.py        # hazard-level simulation (explanatory; circular by construction)
```

---

## Reproducing the real-data validation

Data are **not** included (they are large and governed by their own terms). Download
them yourself as described in `REAL_DATA_STEPS.md`:
- **TCGA-BRCA** (RNA-seq expression + clinical) from UCSC Xena —
  https://xenabrowser.net/datapages/
- **METABRIC** (microarray expression + clinical) from cBioPortal —
  https://www.cbioportal.org/study/summary?id=brca_metabric

**Single cohort (TCGA, leave-one-site-out):**
```bash
python src/xena_validation.py \
  --expr TCGA-BRCA.star_fpkm-uq.tsv.gz --pheno TCGA-BRCA.clinical.tsv.gz \
  --subtype-col "primary_diagnosis.diagnoses" \
  --classes "Infiltrating duct carcinoma, NOS;Lobular carcinoma, NOS" \
  --balanced --topk 500 --min-site 15 --audit 20 --out results.csv
```

**Two cohorts (TCGA + METABRIC, cross-cohort transfer + pooled sites):**
```bash
# 1) one-time: translate TCGA Ensembl IDs to gene symbols (needs: pip install mygene)
python src/make_genemap.py --expr TCGA-BRCA.star_fpkm-uq.tsv.gz --out genemap.tsv

# 2) run the two-cohort audit
python src/twocohort_validation.py \
  --expr1 TCGA-BRCA.star_fpkm-uq.tsv.gz --pheno1 TCGA-BRCA.clinical.tsv.gz \
  --subtype-col1 "primary_diagnosis.diagnoses" \
  --classes1 "Infiltrating duct carcinoma, NOS;Lobular carcinoma, NOS" \
  --expr2 data_mrna_illumina_microarray.txt --pheno2 data_clinical_patient.txt \
  --subtype-col2 "HISTOLOGICAL_SUBTYPE" --classes2 "Ductal/NST;Lobular" \
  --site-col2 "COHORT" --harmonize zscore --gene-map1 genemap.tsv \
  --topk 500 --min-site 15 --audit 20 --out results_twocohort.csv
```
The `real_data/` CSVs are the outputs these commands produced for the paper.

---

## Honest scope and limitations

- Simulation results are illustrative; they are **not** clinical performance.
- The real-data analyses use two breast-cancer cohorts and one binary histological
  task (ductal vs lobular) with a logistic-regression baseline. They are an
  **external-cohort illustration, not a definitive validation**: detector–performance
  correlations are partly confounded by the between-cohort gap (reported via partial
  correlations) and depend on the performance metric.
- Tissue-source site is an administrative label, not a sequencing batch; harmonised
  data understate technical shift relative to a fully independent pipeline.
- We adopt relevant principles of TRIPOD+AI as good practice but **do not claim
  conformance**; DECIDE-AI (live clinical evaluation) is out of scope.

---

## License and citation

MIT License (see `LICENSE`). If you use this code, please cite the paper and the
archived release (see `CITATION.cff`).

**Authors:** Sarthak Saptarshi Bandopadhyay (Independent Researcher, Argyle, Texas, USA; ORCID
0009-0002-4140-8144; sarthaksb@ieee.org) and Manash Chanda (Meghnad Saha Institute
of Technology, Kolkata 700150, India).

**Repository:** https://github.com/sarthaksaptarshi/trace-multiomic-audit

**DOI:** to be added after Zenodo deposit — mint it at deposit time against the exact
version submitted (do not pre-generate it).
