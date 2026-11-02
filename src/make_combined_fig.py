import numpy as np, matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],
 "mathtext.fontset":"stix","font.size":8,"axes.labelsize":8,"axes.titlesize":8.3,
 "xtick.labelsize":6.6,"ytick.labelsize":7,"axes.linewidth":0.8,"legend.frameon":False,
 "savefig.bbox":"tight","savefig.pad_inches":0.02})
C_u="#3B6FB0"; C_b="#E09B2D"; C_c="#C0443E"; GR="#6F6F6F"
fig,(a,b)=plt.subplots(1,2,figsize=(7.1,2.7))
# EXP1
n1=["confidence\nonly","uncertainty\n(4 signals)","build-quality\n(fitted)","combined"]
v1=[0.93,0.91,0.80,0.94]; l1=[0.87,0.84,0.71,0.89]; h1=[0.97,0.96,0.88,0.98]; c1=[C_u,C_u,C_b,C_c]
a.bar(range(4),v1,yerr=[np.array(v1)-l1,np.array(h1)-v1],color=c1,capsize=2,width=0.62)
a.axhline(0.5,ls=(0,(4,2)),lw=0.8,color=GR); a.set_ylim(0,1.02); a.set_xticks(range(4)); a.set_xticklabels(n1)
a.set_ylabel("Failure-detection AUROC"); a.set_title("(a) Realistic shift-driven failures")
a.spines[["top","right"]].set_visible(False)
# EXP2
n2=["confidence","direct\nestimate","provenance","combined\n(conf+prov)"]
v2=[0.45,0.46,0.59,0.65]; l2=[0.33,0.35,0.48,0.54]; h2=[0.56,0.57,0.71,0.76]; c2=[C_u,C_u,C_b,C_c]
b.bar(range(4),v2,yerr=[np.array(v2)-l2,np.array(h2)-v2],color=c2,capsize=2,width=0.62)
b.axhline(0.5,ls=(0,(4,2)),lw=0.8,color=GR); b.set_ylim(0,1.02); b.set_xticks(range(4)); b.set_xticklabels(n2)
b.set_ylabel("Failure-detection AUROC"); b.set_title("(b) Confident-error regime (constructed)")
b.spines[["top","right"]].set_visible(False)
for e in ["png","pdf"]: fig.savefig(f"figures/fig_combined.{e}",dpi=300)
print("wrote fig_combined")
