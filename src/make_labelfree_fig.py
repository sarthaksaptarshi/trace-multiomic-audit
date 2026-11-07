import numpy as np, matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],
 "mathtext.fontset":"stix","font.size":8,"axes.labelsize":8,"axes.titlesize":8.3,
 "xtick.labelsize":6.4,"ytick.labelsize":7,"legend.fontsize":6.4,"axes.linewidth":0.8,
 "lines.linewidth":1.3,"lines.markersize":3.5,"legend.frameon":False,"savefig.bbox":"tight","savefig.pad_inches":0.02})
C1="#4C9A6E";C2="#3B6FB0";C3="#C0443E";GR="#6F6F6F"
import results_io
d=np.load("figures/patient_data.npz")       # written by patient_level.py
PL=results_io.load("patient_level"); PS=results_io.load("prov_sweep")
er=PL["item1"]["ent_shift"]["pearson"]
fig,(a,b,c)=plt.subplots(1,3,figsize=(7.1,2.5))
# (a) surrogate scatter
x=d["ent_shift"]; y=d["dECE"]; a.scatter(x,y,s=10,color=C1,edgecolor="white",linewidth=0.3)
m,q=np.polyfit(x,y,1); xx=np.linspace(x.min(),x.max(),50); a.plot(xx,m*xx+q,color=GR,ls=(0,(4,2)),lw=1)
a.set_xlabel("Entropy shift (label-free)"); a.set_ylabel("True calibration drift $\\Delta$ECE")
a.set_title("(a) Surrogate vs. true drift\n$r=%.2f$ [%.2f, %.2f]"%tuple(er)); a.spines[["top","right"]].set_visible(False)
# (b) baseline AUROC with CI
names=["conf.","entropy","ensemble","domain\nclassif.","TRACE\n(label-free)"]
keys=["predictive confidence","predictive entropy","ensemble disagreement","domain classifier","TRACE trust score (label-free)"]
au=[PL["item2"][k]["auroc"][0] for k in keys]; lo=[PL["item2"][k]["auroc"][1] for k in keys]; hi=[PL["item2"][k]["auroc"][2] for k in keys]
cols=[C2,C2,C2,C2,C1]; xpos=np.arange(5)
b.bar(xpos,au,yerr=[np.array(au)-lo,np.array(hi)-au],color=cols,capsize=2,width=0.62)
b.axhline(0.5,ls=(0,(4,2)),lw=0.8,color=GR); b.set_ylim(0,0.8)
b.set_xticks(xpos); b.set_xticklabels(names); b.set_ylabel("Failure-detection AUROC")
b.set_title("(b) Baselines (shift-driven regime)"); b.spines[["top","right"]].set_visible(False)
# (c) provenance sweep
qn=[p["qnoise"] for p in PS]; ra=[p["auroc"][0] for p in PS]; ral=[p["auroc"][1] for p in PS]; rah=[p["auroc"][2] for p in PS]
rr=[p["r"][0] for p in PS]
c.errorbar(qn,ra,yerr=[np.array(ra)-ral,np.array(rah)-np.array(ra)],fmt='-o',color=C1,capsize=2,label="failure-AUROC")
c.plot(qn,rr,'-s',color=C2,label="$r$(T, ext)")
c.set_xlabel("Provenance misspecification (sd)"); c.set_ylim(0.6,1.02)
c.set_title("(c) Provenance sensitivity"); c.legend(loc="lower left"); c.spines[["top","right"]].set_visible(False)
for e in ["png","pdf"]: fig.savefig(f"figures/fig_labelfree.{e}",dpi=300)
print("wrote fig_labelfree")
