"""Ablation "leaderboard" figure: for each of the 3 mRS targets, rank all 12
configs tried (operating baseline + 11 one-factor-at-a-time variants) by raw
OOF AUROC, with the adopted/operating config highlighted. This is meant to be
more intuitive than a delta-from-zero small-multiples chart: you directly see
where the picked config sits among everything else that was tried, in
absolute performance terms, with a reference line through the baseline's
score so every other bar's shortfall (or rare gain) is visible at a glance.

Usage: python codes/mrs/analysis/make_ablation_leaderboard_figure.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

VIOLET, NEUTRAL = "#4a3aa7", "#9ea6c4"
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
BASE_TAG = "_baseline_REFERENCE_notRerun"
LABEL = {
    BASE_TAG: "Operating baseline",
    "_radam": "RAdam", "_sgd": "SGD",
    "_single_seed": "1 seed instead of 3", "_no_tta": "Test-time augmentation off",
    "_noswa": "Stochastic weight averaging off",
    "_swa_k20": "Weight-averaging window: 20 epochs instead of 10",
    "_uf2": "Unfreeze 2 encoder stages instead of 1",
    "_lighter_reg": "Lighter regularization: dropout 0.3, weight decay 0.01",
    "_higher_enc_lr": "Encoder learning rate x5", "_higher_head_lr": "Head learning rate x3",
    "_no_cosine": "Constant learning rate instead of cosine",
}
TARGET_ORDER = ["mrs3mo", "mrs1y", "dis_mrs"]
TARGET_TITLE = {"mrs3mo": "3-month mRS", "mrs1y": "1-year mRS", "dis_mrs": "Discharge mRS"}

fig, axes = plt.subplots(1, 3, figsize=(19, 6.8))
fig.subplots_adjust(top=0.93, bottom=0.15, left=0.04, right=0.99, wspace=0.95)

for ax, target in zip(axes, TARGET_ORDER):
    sub = df[df.target == target].copy()
    sub = sub.sort_values("auroc", ascending=True).reset_index(drop=True)
    y = np.arange(len(sub))
    base_auroc = sub.loc[sub.tag == BASE_TAG, "auroc"].iloc[0]

    colors = [VIOLET if t == BASE_TAG else NEUTRAL for t in sub.tag]
    xerr = [sub.auroc - sub.ci_lo, sub.ci_hi - sub.auroc]
    ax.barh(y, sub.auroc, color=colors, height=0.62, zorder=3,
            edgecolor=[PRIMARY_INK if t == BASE_TAG else "none" for t in sub.tag], linewidth=1.4)
    ax.errorbar(sub.auroc, y, xerr=xerr, fmt="none", ecolor=PRIMARY_INK,
                elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=4, alpha=0.5)
    ax.axvline(base_auroc, color=VIOLET, linewidth=1.3, linestyle=(0, (3, 2)), zorder=2)

    for yi, (auroc, tag) in enumerate(zip(sub.auroc, sub.tag)):
        w = "bold" if tag == BASE_TAG else "normal"
        ax.text(auroc + 0.006, yi, f"{auroc:.3f}", va="center", ha="left",
                fontsize=8.6, color=PRIMARY_INK, fontweight=w)

    ax.set_yticks(y)
    labels = [LABEL[t] for t in sub.tag]
    ax.set_yticklabels(labels, fontsize=8.8)
    for tick, tag in zip(ax.get_yticklabels(), sub.tag):
        if tag == BASE_TAG:
            tick.set_fontweight("bold"); tick.set_color(VIOLET)

    ax.set_xlim(0.73, 0.875)
    ax.set_xlabel("OOF AUROC (tr-only 5x5x3seed CV)", fontsize=9.5)
    ax.set_title(TARGET_TITLE[target], fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, color=VIOLET), plt.Rectangle((0, 0), 1, 1, color=NEUTRAL)]
fig.legend(handles, ["Adopted (operating baseline)", "Alternative tried"], loc="lower center",
           bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False, fontsize=11.5)

fig.savefig("results/ablation/ablation_leaderboard_figure.png", dpi=1200, bbox_inches="tight")
fig.savefig("results/ablation/ablation_leaderboard_figure.pdf", bbox_inches="tight")
print("saved results/ablation/ablation_leaderboard_figure.{png,pdf}")
