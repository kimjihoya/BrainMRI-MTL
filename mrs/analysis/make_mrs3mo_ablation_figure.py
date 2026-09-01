"""e2e fine-tune hyperparameter sensitivity for mrs3mo (the primary/flagship
target -- the recipe mrs1y/dis_mrs later reused unchanged). Same 11-lever,
tr-only 5x5x3seed CV protocol and delta-vs-baseline small-multiples style as
make_ablation_figure.py's Panel B, kept as its own figure since mrs3mo is the
target the recipe was originally tuned ON (not a "did the reuse generalize"
check like the other two).

Usage: python codes/mrs/analysis/make_mrs3mo_ablation_figure.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

AQUA = "#1baf7a"
SEC_INK, GRID, SURFACE, PRIMARY_INK = "#52514e", "#e1e0d9", "#fcfcfb", "#0b0b0b"

# Helvetica.ttf is not redistributed with this repo (licensed font); drop your
# own copy next to this script to match exactly, else falls back to default sans-serif.
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Helvetica.ttf")
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    FONT_NAME = fm.FontProperties(fname=FONT_PATH).get_name()
else:
    FONT_NAME = "Helvetica"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_NAME],
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": PRIMARY_INK,
    "text.color": PRIMARY_INK,
    "xtick.color": SEC_INK,
    "ytick.color": SEC_INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

df = pd.read_csv("results/ablation/e2e_hparam_results.csv")
df = df[df.target == "mrs3mo"]
BASE_TAG = "_baseline_REFERENCE_notRerun"
base = df[df.tag == BASE_TAG].auroc.iloc[0]


def get(tag):
    return df[df.tag == tag].auroc.iloc[0]


LEVERS = [
    ("Optimizer", [("RAdam", "_radam"), ("SGD", "_sgd")]),
    ("Ensemble size", [("1 seed", "_single_seed")]),
    ("Test-time aug.", [("off", "_no_tta")]),
    ("SWA", [("off", "_noswa")]),
    ("SWA window", [("20 ep", "_swa_k20")]),
    ("Unfreeze depth", [("2 stages", "_uf2")]),
    ("Regularization", [("drop .3 / wd .01", "_lighter_reg")]),
    ("Encoder LR", [("5e-5", "_higher_enc_lr")]),
    ("Head LR", [("3e-3", "_higher_head_lr")]),
    ("LR schedule", [("constant", "_no_cosine")]),
]

fig, axes = plt.subplots(2, 5, figsize=(14, 6.2))
fig.subplots_adjust(top=0.80, bottom=0.14, left=0.06, right=0.98, hspace=0.85, wspace=0.25)
D_LO, D_HI = -0.045, 0.02

for i, (title, alts) in enumerate(LEVERS):
    ax = axes.flat[i]
    xs = np.arange(len(alts))
    deltas = [get(tag) - base for _, tag in alts]
    ax.axhspan(D_LO, 0, color="#f0efec", zorder=0, linewidth=0)
    ax.bar(xs, deltas, width=0.5, color=AQUA, zorder=3)
    for xi, d in zip(xs, deltas):
        ax.text(xi, d + (0.0015 if d >= 0 else -0.0015), f"{d:+.3f}", ha="center",
                va="bottom" if d >= 0 else "top", fontsize=7.2, color=SEC_INK)
    ax.axhline(0, color=PRIMARY_INK, linewidth=1.1, zorder=2)
    ax.set_xlim(-0.6, max(0.6, len(alts) - 0.4))
    ax.set_ylim(D_LO, D_HI)
    ax.set_xticks(xs); ax.set_xticklabels([a[0] for a in alts], fontsize=7.8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5, color=PRIMARY_INK)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    if i % 5 == 0:
        ax.set_ylabel("Delta vs baseline", fontsize=8.8)
        ax.tick_params(axis="y", labelsize=7.8)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

base_patch = plt.Line2D([0], [0], color=PRIMARY_INK, linewidth=1.6)
aqua_patch = plt.Line2D([0], [0], color=AQUA, linewidth=6)
fig.legend([aqua_patch, base_patch], ["3-month mRS (delta vs its own baseline)",
           "0 = operating baseline"], loc="lower center", bbox_to_anchor=(0.5, 0.01),
           ncol=2, frameon=False, fontsize=10.5)

fig.text(0.5, 0.925, "End-to-end fine-tune hyperparameter sensitivity -- 3-month mRS",
          ha="center", va="bottom", fontsize=19, fontweight="bold", color=PRIMARY_INK)
fig.text(0.5, 0.895, "the target this recipe was originally tuned on (baseline AUROC = "
          f"{base:.3f}, tr-only 5x5x3seed CV)", ha="center", va="bottom", fontsize=10.5,
          color=SEC_INK)

fig.savefig("results/ablation/mrs3mo_ablation_figure.png", dpi=1200, bbox_inches="tight")
fig.savefig("results/ablation/mrs3mo_ablation_figure.pdf", bbox_inches="tight")
print("saved results/ablation/mrs3mo_ablation_figure.{png,pdf}")
