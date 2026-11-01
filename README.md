# TRACE — pre-deployment transportability audit (simulation code)

All results in the associated manuscript are produced by this code on
**simulated data only**. No human-subject, clinical, or controlled-access data
are used or required.

## Contents (`src/`)
- `make_figures.py` — OMICS-SIM illustrative cohort figures and the configuration generator.
- `stress_test.py` — adversarial robustness / circularity battery (Table 2, Fig. 5): correlated, nonlinear, noisy, and omitted-mechanism scenarios; design/evaluation split; prespecified failure bar; bootstrap CIs; learned-vs-equal weights.
- `patient_level.py` — patient-level deployment simulator for (i) validating the label-free calibration surrogate and (ii) comparing failure-detection baselines (predictive confidence, entropy, ensemble disagreement, domain classifier) against a fully label-free trust score, with AUROC and AUPRC.
- `prov_sweep.py` — provenance-misspecification sensitivity sweep.
- `make_robust_fig.py`, `make_labelfree_fig.py` — figure generation.

## Reproduce
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/stress_test.py          # robustness battery
python src/patient_level.py        # calibration surrogate + baselines
python src/prov_sweep.py           # provenance sensitivity
python src/make_figures.py         # illustrative figures
python src/make_robust_fig.py
python src/make_labelfree_fig.py
```
Random seeds are fixed; numbers reproduce the manuscript tables/figures.

## Depositing for a citable DOI (author action, before/at submission)
1. Push this folder to a public GitHub repository (fill `repository-code` in `CITATION.cff`).
2. In Zenodo, enable the GitHub repository under *Account → GitHub*, then create a
   GitHub **Release** (e.g., `v1.0.0`). Zenodo mints a DOI automatically from `.zenodo.json`.
3. Paste the minted DOI into `CITATION.cff` (`doi:`) and into the manuscript's
   **Code availability** statement.
4. For double-blind review, provide the Zenodo *reviewer link* or an anonymised repository.

> NOTE: The DOI cannot be created by the authoring tool and must be minted by the
> authors at deposit time. Do not cite a DOI until it exists.
