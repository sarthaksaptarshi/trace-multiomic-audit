"""RUNNABLE real-data validation for TRACE on open-tier TCGA (NOT executed in the
authoring environment, which has no network). Run on a networked machine.

Design: leave-tissue-source-site-out multi-omic subtyping; freeze model,
diagnostics, weights and threshold on source sites; measure external macro-F1 on
held-out site(s); compare TRACE (build-quality), uncertainty baselines, and a
direct estimate from a small labelled target audit; report the COMBINED audit.

IMPORTANT CAVEATS (must be stated in any write-up):
 * GDC harmonises across sequencing centres, so tissue-source site is NOT a
   sequencing centre or processing batch; this tests a weaker shift than an
   independent cohort. Prefer an independent external cohort (e.g. CPTAC/METABRIC,
   access permitting) for a true transportability test.
 * Check subtype-label circularity: if labels are expression-derived (e.g. PAM50),
   do not use expression features to predict them, or hold out the defining assay.
 * Keep target-eval labels sealed until model/diagnostics/threshold are frozen.

Requires: network; `pip install requests pandas numpy scikit-learn`. Open-tier
files need no dbGaP approval. See https://docs.gdc.cancer.gov/API/ .
"""
import requests, io, pandas as pd, numpy as np   # noqa
GDC="https://api.gdc.cancer.gov"

def gdc_files(project, data_type, workflow=None, n=2000):
    """Query open-access files for a project and data type (returns a manifest)."""
    filt={"op":"and","content":[
        {"op":"=","content":{"field":"cases.project.project_id","value":project}},
        {"op":"=","content":{"field":"data_type","value":data_type}},
        {"op":"=","content":{"field":"access","value":"open"}}]}
    if workflow: filt["content"].append({"op":"=","content":{"field":"analysis.workflow_type","value":workflow}})
    params={"filters":filt,"format":"TSV","size":str(n),
            "fields":"file_id,file_name,cases.submitter_id,cases.samples.sample_type,"
                     "cases.tissue_source_site.name,cases.project.project_id"}
    r=requests.post(GDC+"/files",json=params); r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text),sep="\t")

# TODO (author): choose project (e.g. 'TCGA-BRCA'), download STAR-Counts expression,
# methylation beta, copy number, obtain subtype labels from an INDEPENDENT source,
# parse tissue_source_site, build the matrices, then:
#
#   for held_out_site in sites:
#       train on other sites (freeze preprocessing on train only)
#       compute TRACE diagnostics (leakage susceptibility, provenance, stability)
#       compute uncertainty baselines (confidence, entropy, ensemble, domain clf)
#       direct estimate from a small labelled target audit
#       external macro-F1 on held-out site (sealed until frozen)
#   report per-site: each detector's agreement with the realised drop,
#   failure-detection AUROC/AUPRC, and the COMBINED (uncertainty+build-quality) score.
#
# Do NOT report numbers until this has actually been run. No results are included
# in the manuscript because this analysis was not executed by the authoring tool.
if __name__=="__main__":
    raise SystemExit("Configure project/labels, then run on a networked machine.")
