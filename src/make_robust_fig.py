import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],
 "mathtext.fontset":"stix","font.size":8,"axes.labelsize":8,"axes.titlesize":8.5,
 "xtick.labelsize":6.6,"ytick.labelsize":7,"legend.fontsize":6.8,"axes.linewidth":0.8,
 "lines.linewidth":1.3,"lines.markersize":4.5,"legend.frameon":False,
 "savefig.bbox":"tight","savefig.pad_inches":0.02})
C_T="#4C9A6E"; C_B="#3B6FB0"; C_I="#C0443E"; GR="#6F6F6F"
labels=["S0\nidealised","S1\ncorrelated","S2\n+nonlin/\nnoise","S3\n+omit\n(minor)","S4\nomit-\ndominant","S5\nheavy\nnoise"]
import os, results_io; os.makedirs("figures",exist_ok=True)
S=results_io.load("stress_test")            # written by stress_test.py (6 scenarios, in order)
x=np.arange(len(S))
col=lambda k,j:np.array([s[k][j] for s in S])
T,Tl,Th=col("Teq",0),col("Teq",1),col("Teq",2)
B,Bl,Bh=col("best",1),col("best",2),col("best",3)
I=col("intacc_naive",0)
r,rl,rh=col("rT",0),col("rT",1),col("rT",2)
fig,(a,b)=plt.subplots(1,2,figsize=(7.1,2.9))
# panel A: failure-detection AUROC
a.errorbar(x-0.08,T,yerr=[T-Tl,Th-T],fmt='-o',color=C_T,capsize=2,label="Trust score $T$")
a.errorbar(x+0.08,B,yerr=[B-Bl,Bh-B],fmt='-s',color=C_B,capsize=2,label="Best single probe")
a.plot(x,I,'^',color=C_I,label="Internal accuracy (naïve)")
a.axhline(0.5,ls=(0,(4,2)),lw=0.8,color=GR); a.text(5.3,0.52,"chance",fontsize=6,color=GR,ha="right")
a.set_xticks(x); a.set_xticklabels(labels); a.set_ylim(0,1.02)
a.set_ylabel("Failure-detection AUROC"); a.set_title("(a) Detecting deployment failure")
a.legend(loc="lower left"); a.spines[["top","right"]].set_visible(False)
# panel B: Pearson r(T,ext)
b.errorbar(x,r,yerr=[r-rl,rh-r],fmt='-o',color=C_T,capsize=2)
b.axhline(0,ls='-',lw=0.8,color='black')
b.set_xticks(x); b.set_xticklabels(labels); b.set_ylim(0,1.0)
b.set_ylabel("Pearson $r$ (trust score vs. external)")
b.set_title("(b) Predicting external performance")
b.annotate("score fails when a\nmajor driver is\nunmeasured",xy=(4,r[4]),xytext=(2.5,0.22),
 fontsize=6,color=C_I,ha="center",arrowprops=dict(arrowstyle="-|>",color=C_I,lw=0.8))
b.spines[["top","right"]].set_visible(False)
for ext in ["png","pdf"]: fig.savefig(f"figures/fig_robust.{ext}",dpi=300)
print("wrote fig_robust")
