"""Paper Fig. 3 (TCGA + METABRIC) and Fig. 4 (glioma methylation) from the real-data
CSVs plus results/realdata_stats.json (run src/realdata_stats.py with all three CSVs
first). No hard-coded results: bars come from the CSVs, correlations and CIs from the
stats JSON, so the figures match the numbers quoted in the text.

  python src/make_cohort_figs.py
"""
import os, numpy as np, pandas as pd, matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],
 "mathtext.fontset":"stix","font.size":8,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,
 "legend.fontsize":6.6,"axes.linewidth":0.8,"legend.frameon":False,"savefig.bbox":"tight","savefig.pad_inches":0.02})
import results_io; os.makedirs("figures",exist_ok=True)
S=results_io.load("realdata_stats")
TC=pd.read_csv("real_data/results_twocohort.csv")
TX=pd.read_csv("real_data/results_twocohort_crosscohort.csv")
GL=pd.read_csv("real_data/glioma_pilot4.csv")
BLUE,ORANGE,GREY="#3B6FB0","#D9822B","#6F6F6F"
LAB={"leakage":"Leakage susceptibility","provenance":"Provenance","dECE":"Calibration drift",
     "direct":"Direct estimate","stability":"Stability","domain":"Domain classifier",
     "ensemble":"Ensemble disagr.","confidence":"Confidence","entropy":"Entropy"}

def ci_dots(ax,names,stats,key,offset,marker,face,label):
    for i,d in enumerate(names):
        v=stats[d].get(key)
        if not v or v[0] is None: continue
        pt,lo,hi=v
        ax.errorbar(pt,i+offset,xerr=[[pt-lo],[hi-pt]],fmt=marker,ms=4,capsize=2,lw=0.9,
                    color="black",mfc=face,label=label if i==0 else None)

# ---------------- Fig. 3: two cohorts ----------------
fig,axs=plt.subplots(1,3,figsize=(7.2,2.9),gridspec_kw={"width_ratios":[0.7,1.6,1.4]})
ax=axs[0]; lab=[t.replace("->","→\n") for t in TX["transfer"]]
ax.bar(range(len(TX)),TX["AUROC"],color=[BLUE,ORANGE][:len(TX)],width=0.6)
for i,v in enumerate(TX["AUROC"]): ax.text(i,v+0.02,f"{v:.2f}",ha="center",fontsize=7)
ax.set_xticks(range(len(TX))); ax.set_xticklabels(lab,fontsize=6.5); ax.set_ylim(0,1.05)
ax.set_ylabel("External AUROC"); ax.set_title("A  Cross-cohort transfer",loc="left",fontsize=8)

ax=axs[1]; T=TC.sort_values("external_AUROC").reset_index(drop=True)
ax.bar(range(len(T)),T["external_AUROC"],color=[ORANGE if c=="METABRIC" else BLUE for c in T["cohort"]],width=0.75)
ax.set_xticks(range(len(T))); ax.set_xticklabels(T["site"],rotation=90,fontsize=5.5); ax.set_ylim(0,1.05)
ax.set_ylabel("External AUROC (held-out site)")
ax.set_title(f"B  Per-site transportability ({len(T)} pooled sites)",loc="left",fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE,label="TCGA site"),Patch(color=ORANGE,label="METABRIC site")],loc="upper left",bbox_to_anchor=(0,1.0))
ax.set_ylim(0,1.25); ax.set_yticks(np.arange(0,1.01,0.2))

ax=axs[2]; st=S["twocohort"]["external_AUROC"]; names=[d for d in LAB if d in st][::-1]
ci_dots(ax,names,st,"rho",+0.15,"o",BLUE,"raw")
ci_dots(ax,names,st,"partial",-0.15,"D","white","partial (|cohort, balance)")
ax.axvline(0,color=GREY,lw=0.8,ls=(0,(4,2))); ax.set_xlim(-1,1)
ax.set_yticks(range(len(names))); ax.set_yticklabels([LAB[d] for d in names])
ax.set_xlabel(r"Spearman $\rho$ with external AUROC (95% bootstrap CI)")
ax.set_title("C  Predictors of cross-site AUROC",loc="left",fontsize=8)
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.2),ncol=2)
for a_ in axs: a_.spines[["top","right"]].set_visible(False)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(f"figures/fig_twocohort.{e}",dpi=300)
print("wrote figures/fig_twocohort.(png|pdf)")

# ---------------- Fig. 4: glioma ----------------
fig,axs=plt.subplots(1,2,figsize=(7.0,2.6),gridspec_kw={"width_ratios":[1.4,1]})
ax=axs[0]; G=GL.sort_values("external_F1").reset_index(drop=True)
ax.bar(range(len(G)),G["external_F1"],color=BLUE,width=0.7)
ax.set_xticks(range(len(G))); ax.set_xticklabels(G["plate"],rotation=90,fontsize=6.5)
ax.set_ylabel("External macro-F1 (held-out plate-batch)")
ax.set_title(f"A  Per-batch transportability (glioma methylation, {len(G)} batches)",loc="left",fontsize=8)
ax=axs[1]; sg=S["glioma"]["external_F1"]
GLAB={"confidence":"Predictive confidence","entropy":"Predictive entropy","provenance":"Provenance"}
names=[d for d in GLAB if d in sg][::-1]
ci_dots(ax,names,sg,"rho",0,"o",BLUE,None)
ax.axvline(0,color=GREY,lw=0.8,ls=(0,(4,2))); ax.set_xlim(-1,1)
ax.set_yticks(range(len(names))); ax.set_yticklabels([GLAB[d] for d in names])
ax.set_xlabel(r"Spearman $\rho$ with external macro-F1 (95% CI)")
ax.set_title("B  Predictors of cross-batch failure",loc="left",fontsize=8)
for a_ in axs: a_.spines[["top","right"]].set_visible(False)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(f"figures/fig_glioma.{e}",dpi=300)
print("wrote figures/fig_glioma.(png|pdf)")
