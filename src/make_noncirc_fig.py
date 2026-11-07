import numpy as np, matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],
 "mathtext.fontset":"stix","font.size":8,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7.2,
 "legend.fontsize":6.6,"axes.linewidth":0.8,"legend.frameon":False,"savefig.bbox":"tight","savefig.pad_inches":0.02})
import os, results_io; os.makedirs("figures",exist_ok=True)
D=results_io.load("comprehensive")          # written by comprehensive.py
rows=[(nm,*v["auroc"],"trace" if nm.startswith("TRACE") else ("lab" if v["access"]=="LABELS" else "free"))
      for nm,v in D.items()]                # name, auroc, lo, hi, kind
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
