import numpy as np, matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],
 "mathtext.fontset":"stix","font.size":8,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7.2,
 "legend.fontsize":6.6,"axes.linewidth":0.8,"legend.frameon":False,"savefig.bbox":"tight","savefig.pad_inches":0.02})
rows=[  # name, auroc, lo, hi, kind
 ("direct estimate (target-audit labels)",0.93,0.88,0.97,"lab"),
 ("predictive confidence",0.93,0.87,0.97,"free"),
 ("predictive entropy",0.91,0.85,0.96,"free"),
 ("ensemble disagreement",0.86,0.77,0.93,"free"),
 ("calibration drift (audit labels)",0.82,0.72,0.90,"lab"),
 ("stability (source)",0.81,0.71,0.88,"free"),
 ("TRACE (label-dependent)",0.68,0.56,0.79,"trace"),
 ("TRACE (label-free)",0.63,0.51,0.74,"trace"),
 ("domain classifier",0.58,0.45,0.69,"free"),
 ("leakage (source)",0.53,0.41,0.63,"free"),
 ("provenance (source)",0.42,0.30,0.53,"free")]
rows=sorted(rows,key=lambda r:r[1])
y=np.arange(len(rows)); col={"lab":"#3B6FB0","free":"#4C9A6E","trace":"#C0443E"}
fig,ax=plt.subplots(figsize=(6.6,3.4))
for i,(nm,au,lo,hi,k) in enumerate(rows):
    ax.barh(i,au,xerr=[[au-lo],[hi-au]],color=col[k],capsize=2,height=0.62,
            edgecolor="black" if k=="trace" else "none",linewidth=0.8 if k=="trace" else 0)
ax.axvline(0.5,ls=(0,(4,2)),lw=0.8,color="#6F6F6F"); ax.text(0.5,len(rows)-0.3,"chance",fontsize=6,color="#6F6F6F",ha="center")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0,1.0)
ax.set_xlabel("Failure-detection AUROC  (external macro-F1 < design-split median)")
ax.set_title("Non-circular patient-level validation: TRACE is outperformed by simple baselines")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#3B6FB0",label="uses target labels"),Patch(color="#4C9A6E",label="label-free baseline/probe"),Patch(color="#C0443E",label="TRACE composite")],loc="lower right")
ax.spines[["top","right"]].set_visible(False)
for e in ["png","pdf"]: fig.savefig(f"figures/fig_noncirc.{e}",dpi=300)
print("wrote fig_noncirc")
