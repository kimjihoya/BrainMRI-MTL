"""Kaplan-Meier figure: Death & Dementia only (MACE dropped -- no signal,
see SURVIVAL_COX_SUMMARY.md), all 3 mRS targets, styled to match the rest of
this project's figures (Helvetica, dataviz-skill validated ordinal blue ramp).

Grid: 2 rows (Death, Dementia) x 3 columns (3-month / 1-year / discharge mRS,
3-month first since it's the primary target). Reuses the data/grouping logic
from cox_survival_3mrs.py (Ref/Q1/Q2/Q3 risk groups, Youden threshold on
train_cv_oof only) so the curves are guaranteed consistent with cox_results.csv.

Usage: python codes/mrs/analysis/make_km_figure.py   (run from repo root)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from cox_survival_3mrs import (load_outcomes, data_lock, load_probs, youden_threshold,
                                build_surv, assign_quartile_groups, TASKS)

# -- fonts / palette --
# Helvetica.ttf is not redistributed with this repo (licensed font); drop your
# own copy next to this script to match exactly, else falls back to default sans-serif.
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Helvetica.ttf")
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    FONT_NAME = fm.FontProperties(fname=FONT_PATH).get_name()
else:
    FONT_NAME = "Helvetica"

SEC_INK, GRID, SURFACE, PRIMARY_INK = "#52514e", "#e1e0d9", "#fcfcfb", "#0b0b0b"
# validated ordinal ramp (dataviz skill, light->dark blue, single hue, monotone L)
GRP_COLORS = {"Ref(<thr)": "#86b6ef", "Q1": "#3987e5", "Q2": "#1c5cab", "Q3": "#0d366b"}
GRP_ORDER = ["Ref(<thr)", "Q1", "Q2", "Q3"]
GRP_LABEL = {"Ref(<thr)": "Ref (< threshold)", "Q1": "Q1 (low)", "Q2": "Q2 (mid)", "Q3": "Q3 (high)"}

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

TARGET_ORDER = ["mrs3mo", "mrs1y", "dis_mrs"]
TARGET_TITLE = {"mrs3mo": "3-month mRS", "mrs1y": "1-year mRS", "dis_mrs": "Discharge mRS"}
EVENTS = [("Death", "Death", "Death_date"), ("Dementia", "Dementia", "Dementia_date")]


def panel(ax, sdf, thr, show_xlabel):
    g = assign_quartile_groups(sdf, thr)
    present = [l for l in GRP_ORDER if (g["grp"] == l).sum() > 0]
    lr = multivariate_logrank_test(g["t"], g["grp"], g["e"])

    kmf = KaplanMeierFitter()
    for lab in present:
        m = g["grp"] == lab
        kmf.fit(g.loc[m, "t"], g.loc[m, "e"])
        kmf.plot_survival_function(ax=ax, ci_show=False, color=GRP_COLORS[lab],
                                     lw=2.2, legend=False)
    n, ev = len(g), int(g["e"].sum())
    p_str = "log-rank p < 0.001" if lr.p_value < 0.001 else f"log-rank p = {lr.p_value:.3f}"
    ax.text(0.03, 0.04, f"n={n}, events={ev}\n{p_str}",
            transform=ax.transAxes, fontsize=8.3, color=SEC_INK, va="bottom", ha="left")
    ax.set_xlabel("Years from admission" if show_xlabel else "", fontsize=9.5)
    # fixed at 4y: last informative event across every panel is at t=3.79
    # (Death) / t=3.12 (Dementia); nothing happens between there and the
    # cohort max (4.57) so the extra 0.5y of flat line is dropped
    ax.set_xlim(0, 4.0)
    ax.grid(alpha=0.35, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)


def main():
    outc = load_outcomes()
    lock = data_lock(outc)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.1), sharex=False)
    fig.subplots_adjust(top=0.90, bottom=0.13, left=0.07, right=0.98, hspace=0.45, wspace=0.18)

    for col, task in enumerate(TARGET_ORDER):
        fig_label, kr_label, prob_path = TASKS[task]
        mp = load_probs(prob_path)
        thr = youden_threshold(mp)
        d = outc.merge(mp[["record_id", "prob_poor", "split"]], on="record_id", how="inner")

        for row, (ev_name, ecol, dcol) in enumerate(EVENTS):
            ax = axes[row, col]
            sdf, _ = build_surv(d, ecol, dcol, lock)
            panel(ax, sdf, thr, show_xlabel=(row == 1))
            if row == 0:
                ax.set_title(f"{TARGET_TITLE[task]}\n(Ref threshold = {thr:.3f})",
                              fontsize=13, fontweight="bold", pad=10)
            if col == 0:
                ax.set_ylabel(f"{ev_name}-free survival", fontsize=10.5)
            else:
                ax.tick_params(axis="y", labelleft=False)

    # shared legend (risk-group colors)
    handles = [plt.Line2D([0], [0], color=GRP_COLORS[l], lw=2.5) for l in GRP_ORDER]
    fig.legend(handles, [GRP_LABEL[l] for l in GRP_ORDER], loc="lower center",
               bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=False, fontsize=10.5,
               title="mRS-model risk group (Ref threshold per target shown above each column; "
                     "above-threshold patients tertile-split into Q1-Q3)",
               title_fontsize=9)

    fig.savefig("results/cox/km_death_dementia_figure.png", dpi=1200, bbox_inches="tight")
    fig.savefig("results/cox/km_death_dementia_figure.pdf", bbox_inches="tight")
    print("saved results/cox/km_death_dementia_figure.{png,pdf}")


if __name__ == "__main__":
    main()
