import numpy as np, matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],
 "mathtext.fontset":"stix","font.size":8,"axes.labelsize":8,"axes.titlesize":8.3,
 "xtick.labelsize":6.6,"ytick.labelsize":7,"axes.linewidth":0.8,"legend.frameon":False,
 "savefig.bbox":"tight","savefig.pad_inches":0.02})
import os, results_io; os.makedirs("figures",exist_ok=True)
D=results_io.load("combined_variant")       # written by combined_variant.py
C_u="#3B6FB0"; C_b="#E09B2D"; C_c="#C0443E"; GR="#6F6F6F"
fig,(a,b)=plt.subplots(1,2,figsize=(7.1,2.7))
def vals(d,keys): t=[d[k] for k in keys]; return [x[0] for x in t],[x[1] for x in t],[x[2] for x in t]
# EXP1
n1=["confidence\nonly","uncertainty\n(4 signals)","build-quality\n(fitted)","combined"]
v1,l1,h1=vals(D["exp1"],["confidence only","uncertainty (conf+ent+ens+dom)","build-quality (prov+stab+leak)","combined (uncertainty+build-quality)"]); c1=[C_u,C_u,C_b,C_c]
a.bar(range(4),v1,yerr=[np.array(v1)-l1,np.array(h1)-v1],color=c1,capsize=2,width=0.62)
a.axhline(0.5,ls=(0,(4,2)),lw=0.8,color=GR); a.set_ylim(0,1.02); a.set_xticks(range(4)); a.set_xticklabels(n1)
a.set_ylabel("Failure-detection AUROC"); a.set_title("(a) Realistic shift-driven failures")
a.spines[["top","right"]].set_visible(False)
# EXP2
n2=["confidence\n(fixed dir.)","confidence\n(dir.-aware)","direct\nestimate","provenance","combined\n(conf+prov)"]
v2,l2,h2=vals(D["exp2"]["auroc"],["confidence (fixed direction)","direction-aware confidence","direct estimate (audit)","provenance","combined (conf+prov)"]); c2=[C_u,C_u,C_u,C_b,C_c]
b.bar(range(5),v2,yerr=[np.array(v2)-l2,np.array(h2)-v2],color=c2,capsize=2,width=0.62)
b.axhline(0.5,ls=(0,(4,2)),lw=0.8,color=GR); b.set_ylim(0,1.02); b.set_xticks(range(5)); b.set_xticklabels(n2)
b.set_ylabel("Failure-detection AUROC"); b.set_title("(b) Confident-error regime (constructed)")
b.spines[["top","right"]].set_visible(False)
for e in ["png","pdf"]: fig.savefig(f"figures/fig_combined.{e}",dpi=300)
print("wrote fig_combined")
