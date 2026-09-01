"""Publication figure summarizing both ablation studies for the 3 mRS targets:
  (A) modality ablation (clinical / DWI-only / DWI+clinical fusion / our e2e
      model, best hyperparameters from the C/PCA robustness sweep for the
      frozen configs)
  (B) e2e fine-tune hyperparameter sensitivity, small-multiples style (one
      mini-panel per lever, x = values tried, y = OOF AUROC, dashed line =
      the operating baseline) -- mrs1y & dis_mrs, 10 one-factor-at-a-time levers

Palette: dataviz skill's validated categorical palette (light mode).
Usage: python codes/mrs/analysis/make_ablation_figure.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# -- palette (dataviz skill, validated categorical, light mode) --
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
MUTED_INK, SEC_INK, GRID, BASELINE_LINE, SURFACE = (
    "#898781", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb")
PRIMARY_INK = "#0b0b0b"

# Helvetica.ttf is not redistributed with this repo (licensed font). Drop your
# own copy next to this script to match the paper figures exactly; otherwise
# this falls back to matplotlib's default sans-serif.
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Helvetica.ttf")
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    FONT_NAME = fm.FontProperties(fname=FONT_PATH).get_name()
else:
    FONT_NAME = "Helvetica"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_NAME],
    "axes.edgecolor": BASELINE_LINE,
    "axes.labelcolor": PRIMARY_INK,
    "text.color": PRIMARY_INK,
    "xtick.color": SEC_INK,
    "ytick.color": SEC_INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# ============================================================
# Panel A data
# ============================================================
TARGETS = ["mrs3mo", "mrs1y", "dis_mrs"]
TARGET_LABEL = {"dis_mrs": "Discharge mRS", "mrs1y": "1-year mRS", "mrs3mo": "3-month mRS"}

panelA = {
    "dis_mrs":  {"clinical": (0.845, 0.815, 0.872), "dwi": (0.750, 0.716, 0.783), "fusion": (0.845, 0.816, 0.872), "ours": (0.837, 0.803, 0.869)},
    "mrs1y":    {"clinical": (0.832, 0.793, 0.868), "dwi": (0.787, 0.740, 0.832), "fusion": (0.846, 0.811, 0.882), "ours": (0.835, 0.795, 0.873)},
    "mrs3mo":   {"clinical": (0.796, 0.753, 0.835), "dwi": (0.755, 0.714, 0.799), "fusion": (0.813, 0.771, 0.853), "ours": (0.810, 0.766, 0.850)},
}

# ============================================================
# Panel B data: e2e hyperparameter ablation, small multiples
# ============================================================
df = pd.read_csv("results/ablation/e2e_hparam_results.csv")


def get(target, tag):
    r = df[(df.target == target) & (df.tag == tag)].iloc[0]
    return r.auroc, r.ci_lo, r.ci_hi


BASE_TAG = "_baseline_REFERENCE_notRerun"
# (panel title, [(x label, tag), ...])  -- x[0] is always the operating baseline
LEVERS = [
    ("Optimizer", [("AdamW", BASE_TAG), ("RAdam", "_radam"), ("SGD", "_sgd")]),
    ("Ensemble size", [("3 seeds", BASE_TAG), ("1 seed", "_single_seed")]),
    ("Test-time aug.", [("on", BASE_TAG), ("off", "_no_tta")]),
    ("SWA", [("on", BASE_TAG), ("off", "_noswa")]),
    ("SWA window", [("10 ep", BASE_TAG), ("20 ep", "_swa_k20")]),
    ("Unfreeze depth", [("1 stage", BASE_TAG), ("2 stages", "_uf2")]),
    ("Regularization", [("drop .6 / wd .05", BASE_TAG), ("drop .3 / wd .01", "_lighter_reg")]),
    ("Encoder LR", [("1e-5", BASE_TAG), ("5e-5", "_higher_enc_lr")]),
    ("Head LR", [("1e-3", BASE_TAG), ("3e-3", "_higher_head_lr")]),
    ("LR schedule", [("cosine", BASE_TAG), ("constant", "_no_cosine")]),
]

# ============================================================
# Figure
# ============================================================
fig = plt.figure(figsize=(14, 11.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.62)
fig.subplots_adjust(top=0.89, bottom=0.10, left=0.06, right=0.98)

# ---------------------------------------------------------------
# Panel A: grouped bars, "ours" as a 4th bar (not a marker)
# ---------------------------------------------------------------
axA = fig.add_subplot(gs[0])
mod_order = ["clinical", "dwi", "ours"]
mod_label = {"clinical": "Clinical only", "dwi": "DWI only (frozen)",
             "ours": "Our model (e2e fine-tuned)"}
mod_color = {"clinical": BLUE, "dwi": ORANGE, "ours": VIOLET}
n_mod = len(mod_order)
bar_w = 0.24
x = np.arange(len(TARGETS))

for i, mod in enumerate(mod_order):
    vals = [panelA[t][mod][0] for t in TARGETS]
    los = [panelA[t][mod][0] - panelA[t][mod][1] for t in TARGETS]
    his = [panelA[t][mod][2] - panelA[t][mod][0] for t in TARGETS]
    xpos = x + (i - (n_mod - 1) / 2) * bar_w
    ec = PRIMARY_INK if mod == "ours" else SURFACE
    lw = 1.6 if mod == "ours" else 1.2
    axA.bar(xpos, vals, width=bar_w * 0.92, color=mod_color[mod], label=mod_label[mod],
            zorder=3, edgecolor=ec, linewidth=lw)
    axA.errorbar(xpos, vals, yerr=[los, his], fmt="none", ecolor=PRIMARY_INK,
                 elinewidth=1.1, capsize=3, capthick=1.1, zorder=4, alpha=0.55)
    for xi, v in zip(xpos, vals):
        axA.text(xi, v + 0.018, f"{v:.3f}", ha="center", va="bottom", fontsize=8.1, color=SEC_INK)

axA.set_xticks(x); axA.set_xticklabels([TARGET_LABEL[t] for t in TARGETS], fontsize=11)
axA.set_ylabel("OOF AUROC", fontsize=10.5)
axA.set_ylim(0.65, 0.93)
axA.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
axA.set_axisbelow(True)
for spine in ["top", "right"]: axA.spines[spine].set_visible(False)
axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False, fontsize=9.3,
           handletextpad=0.5, columnspacing=1.4)

# ---------------------------------------------------------------
# Panel B: small multiples, one mini-panel per lever.
# y-axis = Delta AUROC vs the operating baseline (NOT raw AUROC) -- this is
# the whole point of the panel: show, per lever, whether deviating from the
# chosen config ever wins. Baseline itself sits at y=0 (the axis line), so
# every bar visible here is "what happens if we don't use the operating
# config". If (almost) every bar sits at/below 0, that IS the justification
# for the baseline choice -- the figure should make that conclusion visible
# at a glance, not require reading 22 CI-overlapping numbers.
# ---------------------------------------------------------------
gsB = gs[1].subgridspec(2, 5, wspace=0.25, hspace=0.85)
D_LO, D_HI = -0.085, 0.025
axesB = []
bar_w2 = 0.32
for i, (title, points) in enumerate(LEVERS):
    r, c = divmod(i, 5)
    ax = fig.add_subplot(gsB[r, c])
    axesB.append(ax)
    alts = points[1:]  # baseline (points[0]) is implicit at y=0
    xs = np.arange(len(alts))
    base = {t: get(t, BASE_TAG)[0] for t in ["mrs1y", "dis_mrs"]}

    ax.axhspan(D_LO, 0, color="#f0efec", zorder=0, linewidth=0)
    for j, (target, color) in enumerate([("mrs1y", BLUE), ("dis_mrs", ORANGE)]):
        deltas = [get(target, tag)[0] - base[target] for _, tag in alts]
        xpos = xs + (j - 0.5) * bar_w2
        ax.bar(xpos, deltas, width=bar_w2 * 0.92, color=color, zorder=3,
               label="1-year mRS" if target == "mrs1y" else "Discharge mRS")
        for xi, d in zip(xpos, deltas):
            ax.text(xi, d + (0.003 if d >= 0 else -0.003), f"{d:+.3f}", ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=6.6, color=SEC_INK)
    ax.axhline(0, color=PRIMARY_INK, linewidth=1.1, zorder=2)
    ax.set_xlim(-0.55, len(alts) - 0.45)
    ax.set_ylim(D_LO, D_HI)
    ax.set_xticks(xs); ax.set_xticklabels([a[0] for a in alts], fontsize=7.8)
    ax.set_title(title, fontsize=9.6, fontweight="bold", pad=5, color=PRIMARY_INK)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    if c == 0:
        ax.set_ylabel("Delta vs baseline", fontsize=8.6)
        ax.tick_params(axis="y", labelsize=7.6)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

# shared legend for panel B (pushed well clear of the bottom row of mini-panels)
handles, labels_ = axesB[0].get_legend_handles_labels()
base_patch = plt.Line2D([0], [0], color=PRIMARY_INK, linewidth=1.6)
fig.legend(handles + [base_patch], labels_ + ["0 = operating baseline"], loc="lower center",
           bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False, fontsize=10.5)

# ---------------------------------------------------------------
# Titles: centered, at the top of each panel block
# ---------------------------------------------------------------
posA0 = axA.get_position()
fig.text(0.5, posA0.y1 + 0.050, "Modality ablation", ha="center", va="bottom",
          fontsize=19, fontweight="bold", color=PRIMARY_INK)

topB = max(ax.get_position().y1 for ax in axesB)
fig.text(0.5, topB + 0.062, "End-to-end fine-tune hyperparameter sensitivity", ha="center",
          va="bottom", fontsize=19, fontweight="bold", color=PRIMARY_INK)

# panel corner labels "A" / "B"
posA, posB = axA.get_position(), axesB[0].get_position()
fig.text(posA.x0 - 0.045, posA0.y1 + 0.085, "A", transform=fig.transFigure,
          fontsize=20, fontweight="bold", color=PRIMARY_INK, ha="left", va="bottom")
fig.text(posB.x0 - 0.045, topB + 0.095, "B", transform=fig.transFigure,
          fontsize=20, fontweight="bold", color=PRIMARY_INK, ha="left", va="bottom")

fig.savefig("results/ablation/ablation_summary_figure.png", dpi=1200, bbox_inches="tight")
fig.savefig("results/ablation/ablation_summary_figure.pdf", bbox_inches="tight")
print("saved results/ablation/ablation_summary_figure.{png,pdf}")
