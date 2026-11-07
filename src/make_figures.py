#!/usr/bin/env python3
"""
Figure generator for the TRACE audit-framework manuscript.
Produces 12 figures as 300-dpi PNG and vector PDF.

PROVENANCE OF THE PLOTTED VALUES -- read before using any of these in a paper:
  * Diagrams (no data):                       fig1, fig2
  * Simulated, generated here from a seed:    fig8 (generate_configs; circular hazard model)
  * SCHEMATIC -- curves/distributions drawn from hand-chosen parameters or closed-form
    formulas, NOT produced by any analysis:  fig5, fig6, fig9, fig11
    (fig11 also conflicts with src/extras.py, which finds no within-deployment deferral gain)
  * Hard-coded values with NO generating code in this repository:
                                              fig3, fig4, fig7, fig10, fig12
Schematic figures are stamped "Schematic - illustrative values" by default
(STAMP_SCHEMATIC below). Results figures belong in make_*_fig.py, which read
results/*.json written by the analysis scripts.
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Global style: Times-like serif (STIX), tight sizes for column-width figures.
# ----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "axes.grid": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Palette (colour-blind safe, prints well in grayscale)
C_INT   = "#3B6FB0"   # internal / blue
C_EXT   = "#C0443E"   # external / red
C_G1    = "#4C9A6E"   # green
C_G2    = "#E09B2D"   # amber
C_G3    = "#7E6AA8"   # purple
C_GRAY  = "#6F6F6F"
COL1 = 3.45   # single-column width (in)
COL2 = 7.10   # double-column width (in)

STAMP_SCHEMATIC = True
SCHEMATIC = {"fig5_leakage", "fig6_calibration", "fig9_shortcut_curve", "fig11_deferral"}
NO_SOURCE = {"fig3_modalities", "fig4_shortcut_repair", "fig7_stability",
             "fig10_smallsample", "fig12_casestudy"}

def save(fig, name):
    if STAMP_SCHEMATIC and name in SCHEMATIC:
        fig.text(0.99, 0.01, "Schematic - illustrative values, not results",
                 ha="right", va="bottom", fontsize=5.5, color=C_GRAY, style="italic")
    if name in NO_SOURCE:
        print(f"WARNING: {name} plots hard-coded values with no generating code in this repo")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------------
# Shared, reproducible generator for the 48 model configurations.
# Used by Fig. 8 (circular hazard model; see stress_test.py for the analysed version).
# ----------------------------------------------------------------------------
def generate_configs(seed=11, n=48):
    r = np.random.default_rng(seed)
    # four independent failure hazards in [0,1], one per probe:
    #   h0 leakage, h1 low provenance (batch/shortcut features),
    #   h2 selection instability, h3 calibration stress.
    h = r.uniform(0, 1, (4, n))
    extF1 = (0.97 - 0.17*h.sum(0) + r.normal(0, 0.020, n)).clip(0.25, 0.97)
    Lh   = (h[0] + r.normal(0, 0.05, n)).clip(0, 1)        # leakage score (bad high)
    Q    = (1 - h[1] + r.normal(0, 0.05, n)).clip(0, 1)    # provenance (good high)
    stab = (1 - h[2] + r.normal(0, 0.05, n)).clip(0, 1)    # stability (good high)
    cal  = (h[3] + r.normal(0, 0.05, n)).clip(0, 1)        # calib drift (bad high)
    T = (0.25*((1-Lh) + Q + stab + (1-cal))).clip(0, 1)
    # internal accuracy is inflated by the very hazards that break transport
    intAcc = (0.90 + 0.06*h[0] + 0.05*h[1] + r.normal(0, 0.03, n)).clip(0.6, 0.995)
    return dict(extF1=extF1, intAcc=intAcc, Lh=Lh, Q=Q, stab=stab, cal=cal, T=T)

# ============================================================================
# FIG 1 — Framework overview (TRACE audit pipeline)
# ============================================================================
def fig1():
    fig, ax = plt.subplots(figsize=(COL2, 2.95))
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=7.6, tc="black"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                           linewidth=1.0, edgecolor=ec, facecolor=fc)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=9, linewidth=1.0, color=C_GRAY))

    # inputs
    box(1, 29, 17, 9, "Trained model\n+ held-out site data", "#EAF0F7", C_INT)
    # four probes
    probes = [
        (24, 34.5, "Leakage probe", "#F6E7E6", C_EXT),
        (24, 23.5, "Provenance  $\\bar{q}$", "#EAF3EE", C_G1),
        (24, 12.5, "Stability  (SAFE)", "#F3EFE2", C_G2),
        (24, 1.5,  "Calibration drift", "#EEEAF3", C_G3),
    ]
    for (x, y, t, fc, ec) in probes:
        box(x, y, 20, 8.5, t, fc, ec)
        arrow(18.2, 33.5, 23.6, y + 4.25)
    # trust score
    box(50, 18, 17, 10, "Trust score\n$T\\in[0,1]$", "#FCF3D6", "#B08A17", fs=8)
    for (_, y, _, _, _) in probes:
        arrow(44.2, y + 4.25, 49.6, 23)
    # decision
    box(73, 27, 12, 8.5, "Deploy", "#E6F2E9", C_G1)
    box(73, 9,  12, 8.5, "Defer to\nexpert", "#F6E7E6", C_EXT)
    arrow(67.2, 24.5, 72.6, 31.0)
    arrow(67.2, 21.5, 72.6, 13.5)
    ax.text(70.3, 32.2, "$T\\geq\\tau$", fontsize=7, color=C_G1)
    ax.text(70.1, 11.3, "$T<\\tau$", fontsize=7, color=C_EXT)
    # post-deploy monitor
    box(88, 18, 11, 10, "Monitor\non new sites", "#F0F0F0", C_GRAY, fs=7)
    arrow(85.2, 31.2, 87.6, 28.4)
    # continual-monitoring feedback loop, routed up and over the top row
    ax.add_patch(FancyArrowPatch((93.5, 28.2), (9.0, 38.2), arrowstyle="-|>",
                 mutation_scale=9, linewidth=1.0, color=C_GRAY,
                 connectionstyle="arc3,rad=0.33", linestyle=(0,(4,2))))
    ax.text(52, 49.0, "continual monitoring", fontsize=6.3, color=C_GRAY,
            ha="center", style="italic",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    save(fig, "fig1_framework")

# ============================================================================
# FIG 2 — OMICS-SIM cohort design
# ============================================================================
def fig2():
    fig, ax = plt.subplots(figsize=(COL2, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    blocks = ["Mutation", "Methylation", "Expression", "CNV", "miRNA"]
    bw = [10, 16, 20, 11, 8]            # relative block widths
    x0 = 10.0
    xs = []
    x = x0
    for w in bw:
        xs.append((x, w)); x += w + 1.2
    total_right = x

    centres = [("Centre A", 0, "#DCE7F3"), ("Centre B", 1, "#DCE7F3"),
               ("Centre C (external)", 2, "#F6E1E0")]
    rows_y = [32, 22, 9.5]
    row_h = [8.5, 8.5, 11.5]
    npat = ["n = 180", "n = 180", "n = 160"]

    # data-matrix blocks
    for ci, (cname, _, fc) in enumerate(centres):
        y = rows_y[ci]; h = row_h[ci]
        ax.text(x0 - 1.5, y + h/2, cname, ha="right", va="center", fontsize=7.2)
        ax.text(x0 - 1.5, y + h/2 - 2.6, npat[ci], ha="right", va="center",
                fontsize=6.3, color=C_GRAY)
        for (bx, bwi) in xs:
            ax.add_patch(Rectangle((bx, y), bwi, h, facecolor=fc,
                         edgecolor="white", linewidth=0.8))
    # shortcut overlay on methylation (array batch, strongest at Centre C)
    mx, mw = xs[1]
    for ci in range(3):
        y = rows_y[ci]; h = row_h[ci]
        alpha = [0.18, 0.30, 0.6][ci]
        ax.add_patch(Rectangle((mx, y), mw, h, facecolor="#C0443E",
                     edgecolor="none", alpha=alpha, hatch="////"))
    # block labels
    for (bx, bwi), name in zip(xs, blocks):
        ax.text(bx + bwi/2, 42.5, name, ha="center", va="bottom", fontsize=6.8)
        ax.text(bx + bwi/2, 41.0, f"$p_b$", ha="center", va="top", fontsize=6,
                color=C_GRAY)
    # module ticks (12 planted modules spread across blocks)
    mod_per = [2, 2, 4, 2, 2]
    for (bx, bwi), m in zip(xs, mod_per):
        for k in range(m):
            cx = bx + bwi*(k+1)/(m+1)
            ax.plot([cx, cx], [40.4, 41.2], color="black", lw=0.8)
    # legend / annotations
    ax.plot([], [])
    ax.text(50, 3.2, "12 planted signal modules (ticks)   •   "
            "array-batch shortcut injected into Methylation (red overlay; "
            "darker = stronger, peaking at the external site)",
            ha="center", va="center", fontsize=6.3, color="black")
    ax.text(2, 45.2, "520 patients  ·  4 molecular subtypes  ·  3 centres  "
            "(A, B train — C held out)", fontsize=7.2, ha="left")
    save(fig, "fig2_omicssim")

# ============================================================================
# FIG 3 — Internal vs external macro-F1 by modality and fusion
# ============================================================================
def fig3():
    labels = ["Mutation", "Methyl.", "Express.", "CNV", "Early\nfusion", "PW-\nFusion"]
    internal = [0.74, 0.508, 0.71, 0.58, 0.962, 0.951]
    external = [0.70, 0.285, 0.66, 0.52, 0.951, 0.946]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    b1 = ax.bar(x - w/2, internal, w, label="Internal (A+B)", color=C_INT)
    b2 = ax.bar(x + w/2, external, w, label="External (C)", color=C_EXT)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Macro-F1")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.axhline(0.42, ls=(0,(4,2)), lw=0.8, color=C_GRAY)
    ax.text(3.5, 0.445, "majority", fontsize=5.8, color=C_GRAY,
            ha="center", va="bottom")
    # highlight methylation collapse (text sits in the empty band above methyl)
    ax.annotate("array-batch\ncollapse", xy=(1+w/2, 0.285), xytext=(0.98, 0.735),
                fontsize=6, color=C_EXT, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=C_EXT, lw=0.8))
    ax.legend(loc="upper left", ncol=1, bbox_to_anchor=(0.0, 1.03))
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig3_modalities")

# ============================================================================
# FIG 4 — Shortcut gap and provenance-weighted repair (slope chart)
# ============================================================================
def fig4():
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    xpos = [0, 1]
    unw = [0.508, 0.285]
    qw  = [0.508, 0.422]
    ax.plot(xpos, unw, "-o", color=C_EXT, label="Unweighted")
    ax.plot(xpos, qw, "-s", color=C_G1, label="Provenance-weighted")
    ax.set_xlim(-0.35, 1.5)
    ax.set_xticks(xpos); ax.set_xticklabels(["Internal\n(A+B)", "External\n(C)"])
    ax.set_ylabel("Methylation macro-F1")
    ax.set_ylim(0.2, 0.56)
    ax.annotate("", xy=(1.06, 0.285), xytext=(1.06, 0.508),
                arrowprops=dict(arrowstyle="<->", color=C_EXT, lw=0.8))
    ax.text(1.10, 0.396, "gap\n0.223", fontsize=6, color=C_EXT, va="center")
    ax.annotate("", xy=(1.30, 0.422), xytext=(1.30, 0.508),
                arrowprops=dict(arrowstyle="<->", color=C_G1, lw=0.8))
    ax.text(1.34, 0.465, "gap\n0.086", fontsize=6, color=C_G1, va="center")
    ax.legend(loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig4_shortcut_repair")

# ============================================================================
# FIG 5 — Leakage probe: permuted-label accuracy distributions
# ============================================================================
def fig5():
    proper = rng.normal(0.42, 0.045, 2000).clip(0.20, 0.62)
    leaked = rng.normal(0.80, 0.040, 2000).clip(0.63, 0.94)
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    bins = np.linspace(0.15, 0.98, 46)
    ax.hist(proper, bins=bins, color=C_INT, alpha=0.85,
            label="Selection inside CV")
    ax.hist(leaked, bins=bins, color=C_EXT, alpha=0.85,
            label="Selection outside CV")
    ax.axvline(0.25, ls=(0,(4,2)), lw=0.8, color=C_GRAY)
    ax.text(0.25, ax.get_ylim()[1]*0.96, " chance 0.25", fontsize=6,
            color=C_GRAY, va="top")
    ax.set_xlabel("Accuracy under permuted labels")
    ax.set_ylabel("Count")
    ax.legend(loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig5_leakage")

# ============================================================================
# FIG 6 — Calibration reliability diagrams (external site)
# ============================================================================
def fig6():
    conf = np.linspace(0.05, 0.95, 10)
    # early fusion: over-confident (accuracy below diagonal)
    acc_early = conf - 0.16*np.sin(np.pi*conf) - 0.04
    acc_early = acc_early.clip(0, 1)
    # PW-Fusion: closer to diagonal
    acc_pw = conf - 0.09*np.sin(np.pi*conf) + 0.01
    acc_pw = acc_pw.clip(0, 1)
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    ax.plot([0, 1], [0, 1], color=C_GRAY, lw=0.9, ls=(0,(3,2)),
            label="Perfect")
    ax.plot(conf, acc_early, "-o", color=C_EXT,
            label="Early fusion (ECE 0.184)")
    ax.plot(conf, acc_pw, "-s", color=C_G1,
            label="PW-Fusion (ECE 0.123)")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig6_calibration")

# ============================================================================
# FIG 7 — Stability selection: Jaccard, module coverage, artefacts
# ============================================================================
def fig7():
    methods = ["Lasso", "Stability", "RF", "SAFE"]
    jacc = [0.11, 0.18, 0.44, 0.33]
    cover = [4, 5, 5, 9]           # of 12 modules
    artefacts = [6, 4, 3, 0]       # in top-50
    x = np.arange(len(methods))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(COL2, 2.5))
    cols = [C_GRAY, C_G3, C_G2, C_G1]
    axA.bar(x, jacc, 0.6, color=cols)
    axA.set_xticks(x); axA.set_xticklabels(methods)
    axA.set_ylabel("Split-half Jaccard@50")
    axA.set_ylim(0, 0.5)
    for xi, v in zip(x, jacc):
        axA.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.3)
    axA.set_title("(a) Selection reproducibility")
    axA.spines[["top", "right"]].set_visible(False)

    w = 0.38
    axB.bar(x - w/2, cover, w, color=C_G1, label="Modules recovered (/12)")
    axB.bar(x + w/2, artefacts, w, color=C_EXT, label="Artefacts in top-50")
    axB.set_xticks(x); axB.set_xticklabels(methods)
    axB.set_ylabel("Count")
    axB.set_ylim(0, 12)
    axB.axhline(12, ls=(0,(2,2)), lw=0.7, color=C_GRAY)
    axB.legend(loc="upper left")
    axB.set_title("(b) What the top-50 contains")
    axB.spines[["top", "right"]].set_visible(False)
    save(fig, "fig7_stability")

# ============================================================================
# FIG 8 — HEADLINE: trust score tracks external F1; internal accuracy does not
# ============================================================================
def fig8():
    d = generate_configs()
    T, extF1, intAcc = d["T"], d["extF1"], d["intAcc"]
    rT = np.corrcoef(T, extF1)[0,1]
    rI = np.corrcoef(intAcc, extF1)[0,1]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(COL2, 2.7))
    axA.scatter(T, extF1, s=18, color=C_G1, edgecolor="white", linewidth=0.4)
    m, b = np.polyfit(T, extF1, 1)
    xx = np.linspace(T.min(), T.max(), 50)
    axA.plot(xx, m*xx+b, color=C_GRAY, lw=1.0, ls=(0,(4,2)))
    axA.set_xlabel("Trust score $T$")
    axA.set_ylabel("External macro-F1")
    axA.set_title(f"(a) Trust vs. external   $r={rT:.2f}$")
    axA.set_ylim(0.2, 1.0)
    axA.spines[["top","right"]].set_visible(False)

    axB.scatter(intAcc, extF1, s=18, color=C_INT, edgecolor="white", linewidth=0.4)
    m2, b2 = np.polyfit(intAcc, extF1, 1)
    xx2 = np.linspace(intAcc.min(), intAcc.max(), 50)
    axB.plot(xx2, m2*xx2+b2, color=C_GRAY, lw=1.0, ls=(0,(4,2)))
    axB.set_xlabel("Internal accuracy")
    axB.set_ylabel("External macro-F1")
    axB.set_title(f"(b) Internal vs. external   $r={rI:.2f}$")
    axB.set_ylim(0.2, 1.0)
    axB.spines[["top","right"]].set_visible(False)
    save(fig, "fig8_trust_vs_external")

# ============================================================================
# FIG 9 — Shortcut-injection degradation with/without provenance weighting
# ============================================================================
def fig9():
    s = np.linspace(0, 1, 11)
    unw = (0.95 - 0.60*s**1.1 + rng.normal(0,0.004,len(s)))
    qw  = (0.95 - 0.16*s**1.3 + rng.normal(0,0.004,len(s)))
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    ax.plot(s, unw, "-o", color=C_EXT, label="Unweighted")
    ax.plot(s, qw, "-s", color=C_G1, label="Provenance-weighted")
    ax.fill_between(s, unw, qw, color=C_G1, alpha=0.08)
    ax.set_xlabel("Injected shortcut strength")
    ax.set_ylabel("External macro-F1")
    ax.set_ylim(0.25, 1.0)
    ax.legend(loc="lower left")
    ax.spines[["top","right"]].set_visible(False)
    save(fig, "fig9_shortcut_curve")

# ============================================================================
# FIG 10 — Small-sample robustness
# ============================================================================
def fig10():
    n = np.array([40, 80, 160, 240, 360])
    pw    = np.array([0.81, 0.88, 0.92, 0.935, 0.946])
    early = np.array([0.67, 0.79, 0.88, 0.925, 0.951])
    expr  = np.array([0.42, 0.52, 0.60, 0.65, 0.69])
    fig, ax = plt.subplots(figsize=(COL1, 2.45))
    ax.plot(n, pw, "-o", color=C_G1, label="PW-Fusion")
    ax.plot(n, early, "-s", color=C_INT, label="Early fusion")
    ax.plot(n, expr, "-^", color=C_G2, label="Expression only")
    ax.set_xlabel("Training-set size")
    ax.set_ylabel("External macro-F1")
    ax.set_ylim(0.35, 1.0)
    ax.annotate("n=40", xy=(40, 0.81), xytext=(70, 0.58),
                fontsize=6, color=C_GRAY,
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.7))
    ax.legend(loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    save(fig, "fig10_smallsample")

# ============================================================================
# FIG 11 — Deferral: risk-coverage curves
# ============================================================================
def fig11():
    cov = np.linspace(1.0, 0.3, 30)
    # selective error among answered cases; both equal the base error at full
    # coverage and fall as coverage shrinks, trust-guided faster.
    err_conf  = 0.054 * (0.45 + 0.55*cov)
    err_trust = 0.054 * (0.12 + 0.88*cov)
    fig, ax = plt.subplots(figsize=(COL1, 2.45))
    ax.plot(cov, err_conf, "-o", color=C_INT, label="Confidence only")
    ax.plot(cov, err_trust, "-s", color=C_G1, label="Trust-guided")
    ax.set_xlabel("Coverage (fraction answered)")
    ax.set_ylabel("Error among answered")
    ax.set_xlim(1.02, 0.28)   # reverse: full coverage at left
    ax.set_ylim(0, 0.065)
    ax.legend(loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    save(fig, "fig11_deferral")

# ============================================================================
# FIG 12 — Index patient case study (external site)
# ============================================================================
def fig12():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(COL2, 2.5))
    subs = ["A", "B", "C", "D"]
    probs = [0.07, 0.06, 0.82, 0.05]
    cols = [C_GRAY, C_GRAY, C_G1, C_GRAY]
    axA.bar(subs, probs, color=cols, width=0.62)
    axA.set_ylabel("Predicted probability")
    axA.set_xlabel("Molecular subtype")
    axA.set_ylim(0, 1.0)
    axA.text(2, 0.84, "0.82", ha="center", fontsize=6.5, color=C_G1)
    axA.set_title("(a) Subtype posterior")
    axA.spines[["top","right"]].set_visible(False)

    blocks = ["Mutation\n(MMR)", "Methyl.", "Express.", "CNV", "miRNA"]
    contrib = [2.01, 0.66, 0.30, 0.10, -0.05]
    bc = [C_G1 if v >= 0 else C_EXT for v in contrib]
    y = np.arange(len(blocks))[::-1]
    axB.barh(y, contrib, color=bc, height=0.6)
    axB.set_yticks(y); axB.set_yticklabels(blocks)
    axB.set_xlabel("Log-odds contribution to subtype C")
    axB.axvline(0, color="black", lw=0.8)
    for yi, v in zip(y, contrib):
        axB.text(v + (0.06 if v >= 0 else -0.06), yi, f"{v:+.2f}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=6.2)
    axB.set_xlim(-0.6, 2.5)
    axB.set_title("(b) Per-block attribution")
    axB.spines[["top","right"]].set_visible(False)
    save(fig, "fig12_casestudy")

for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12]:
    f()
print("ALL FIGURES DONE")
