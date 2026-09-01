"""Cox + Kaplan-Meier survival analysis for the 3 mRS-model poor-outcome
probabilities (dis_mrs, mrs1y, mrs3mo) against Death / Dementia / MACE.

BRAIN internal cohort ONLY (train_cv_oof + holdout_test). Asan external-validation
cohort has no Death/Dementia/MACE follow-up in this project and is excluded.

Index date = arrival (admission). Event -> event_date. Censored -> administrative
data-lock (latest observed event date anywhere in the cohort). Each prob_poor is
the out-of-fold (train-CV) / holdout prediction from its operating e2e model, so
a patient's own outcome was never used to score that patient.

High/low risk-group threshold = Youden-optimal cut computed on train_cv_oof ONLY
(honest, holdout-blind), same convention as results/mrs/mrs_confusion_matrix.csv.

Outputs (all under results/cox/):
  cox_results.csv        every fitted Cox term (task x outcome x model x covariate)
  km_<task>.png           3-panel KM figure (Death / Dementia / MACE) per task
  SURVIVAL_COX_SUMMARY.md human-readable summary with embedded tables

Usage: python codes/mrs/analysis/cox_survival_3mrs.py   (run from repo root)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.metrics import roc_curve

OUT = "data/csvs/outcomes_final_v2.csv"
RESULT_DIR = "results/cox"

TASKS = {
    # key: (label for figures/ASCII-safe, Korean label for markdown, prob csv)
    "dis_mrs":  ("discharge mRS>=3", "퇴원 mRS>=3",  "results/dismrs_e2e/dismrs_probabilities_by_id.csv"),
    "mrs1y":    ("1y mRS>=3",        "1년 mRS>=3",   "results/mrs1y_e2e/mrs1y_probabilities_by_id.csv"),
    "mrs3mo":   ("3mo mRS>=3",       "3개월 mRS>=3", "results/mrs/mrs_probabilities_by_id.csv"),
}
EVENTS = [
    ("Death",    "Death",    "Death_date"),
    ("Dementia", "Dementia", "Dementia_date"),
    ("MACE",     "MACE",     "MACE_date"),
]
COL = {"Death": "#d7191c", "Dementia": "#fdae61", "MACE": "#2c7bb6"}


def load_outcomes():
    df = pd.read_csv(OUT, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    df["arrival"] = pd.to_datetime(df["arrival"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["male"] = (df["male"].astype(str).str.lower() == "m").astype(int)
    return df


def data_lock(df):
    dates = [pd.to_datetime(df[c], errors="coerce") for c in
             ["Death_date", "Dementia_date", "MACE_date"]]
    return pd.concat(dates).max()


def load_probs(path):
    mp = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    mp.columns = [c.strip().lstrip("﻿") for c in mp.columns]
    mp["prob_poor"] = pd.to_numeric(mp["prob_poor"], errors="coerce")
    mp["y_true"] = pd.to_numeric(mp["y_true"], errors="coerce")
    return mp[["record_id", "prob_poor", "y_true", "split"]]


def youden_threshold(mp):
    """Honest cut: Youden's J on train_cv_oof only (holdout never touched)."""
    oof = mp[mp["split"] == "train_cv_oof"]
    fpr, tpr, thr = roc_curve(oof["y_true"], oof["prob_poor"])
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def build_surv(d, event_col, date_col, lock):
    ev = (pd.to_numeric(d[event_col], errors="coerce") == 1).astype(int)
    edate = pd.to_datetime(d[date_col], errors="coerce")
    end = edate.where(ev == 1, lock)
    days = (end - d["arrival"]).dt.days.astype(float)
    out = pd.DataFrame({
        "t": days / 365.25, "e": ev, "prob_poor": d["prob_poor"],
        "age": d["age"], "male": d["male"], "split": d["split"],
    })
    bad = (~np.isfinite(out["t"])) | (out["t"] <= 0)
    out = out[~bad & out["prob_poor"].notna()].copy()
    return out, int(bad.sum())


def fit_rows(task, ev_name, sdf, covars, model_label, rows):
    cph = CoxPHFitter()
    cph.fit(sdf[covars + ["t", "e"]].dropna(), duration_col="t", event_col="e")
    s = cph.summary
    n, nev, c = len(sdf), int(sdf["e"].sum()), cph.concordance_index_
    for cov in covars:
        rows.append({
            "task": task, "outcome": ev_name, "model": model_label, "covariate": cov,
            "n": n, "n_events": nev, "c_index": round(c, 3),
            "HR": s.loc[cov, "exp(coef)"], "HR_lo": s.loc[cov, "exp(coef) lower 95%"],
            "HR_hi": s.loc[cov, "exp(coef) upper 95%"], "p": s.loc[cov, "p"],
        })
    return cph


GRP_COLORS = {"Ref(<thr)": "#2c7bb6", "Q1": "#abd9e9", "Q2": "#fdae61", "Q3": "#d7191c"}


def assign_quartile_groups(sdf, thr):
    """Ref = below Youden threshold. Above-threshold patients split into
    tertiles (Q1=lowest risk third of the high-risk group ... Q3=highest)."""
    g = sdf.copy()
    above = g["prob_poor"] >= thr
    g["grp"] = "Ref(<thr)"
    if above.sum() >= 3:
        g.loc[above, "grp"] = pd.qcut(g.loc[above, "prob_poor"], 3, labels=["Q1", "Q2", "Q3"])
    else:
        g.loc[above, "grp"] = "Q3"  # too few to split into 3, dump into top bin
    g["grp"] = pd.Categorical(g["grp"], categories=["Ref(<thr)", "Q1", "Q2", "Q3"])
    return g


def km_panel(ax, sdf, thr, ev_name, task_label):
    g = assign_quartile_groups(sdf, thr)
    labels_present = [l for l in ["Ref(<thr)", "Q1", "Q2", "Q3"] if (g["grp"] == l).sum() > 0]

    # Cox: dummy-coded groups, reference = Ref(<thr)
    dummies = pd.get_dummies(g["grp"]).astype(float)
    risk_labels = [l for l in labels_present if l != "Ref(<thr)"]
    group_stats = {}
    if len(risk_labels) > 0:
        gg = pd.concat([g[["t", "e"]], dummies[risk_labels]], axis=1)
        cph = CoxPHFitter()
        cph.fit(gg, "t", "e")
        s = cph.summary
        for lab in risk_labels:
            group_stats[lab] = (s.loc[lab, "exp(coef)"], s.loc[lab, "exp(coef) lower 95%"],
                                 s.loc[lab, "exp(coef) upper 95%"], s.loc[lab, "p"])
    lr = multivariate_logrank_test(g["t"], g["grp"], g["e"])

    kmf = KaplanMeierFitter()
    for lab in labels_present:
        m = g["grp"] == lab
        kmf.fit(g.loc[m, "t"], g.loc[m, "e"],
                label=f"{lab} (n={int(m.sum())}, ev={int(g.loc[m,'e'].sum())})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=GRP_COLORS[lab], lw=2)
    ttl = f"{task_label} -> {ev_name}\nlog-rank p={lr.p_value:.1e}"
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel("Years from admission")
    ax.set_ylabel("Event-free survival")
    ax.set_xlim(0, g["t"].quantile(0.98))
    ax.legend(fontsize=8, loc="lower left", title=f"thr={thr:.3f}")
    ax.grid(alpha=0.25)

    out = {"logrank_p": lr.p_value}
    for lab in ["Ref(<thr)", "Q1", "Q2", "Q3"]:
        m = g["grp"] == lab
        out[f"n_{lab}"] = int(m.sum())
        out[f"ev_{lab}"] = int(g.loc[m, "e"].sum())
        if lab in group_stats:
            hr, lo, hi, p = group_stats[lab]
            out[f"HR_{lab}"], out[f"HR_lo_{lab}"], out[f"HR_hi_{lab}"], out[f"p_{lab}"] = hr, lo, hi, p
        else:
            out[f"HR_{lab}"] = out[f"HR_lo_{lab}"] = out[f"HR_hi_{lab}"] = out[f"p_{lab}"] = np.nan
    return out


def main():
    outc = load_outcomes()
    lock = data_lock(outc)
    print(f"data-lock (censor) = {lock.date()}")

    cox_rows, km_rows = [], []
    md_sections = []

    for task, (fig_label, kr_label, prob_path) in TASKS.items():
        mp = load_probs(prob_path)
        thr = youden_threshold(mp)
        d = outc.merge(mp[["record_id", "prob_poor", "split"]], on="record_id", how="inner")
        print(f"\n=== {task} ({kr_label})  n={len(d)}  Youden thr(OOF)={thr:.3f} ===")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        md_sections.append(f"\n## {task} — {kr_label}\n\n"
                            f"- 확률 소스: `{prob_path}`\n"
                            f"- 코호트 n={len(d)}, 기준(Ref) 컷(Youden, train_cv_oof only) = **{thr:.3f}**"
                            f" — 컷 미만은 Ref, 이상인 환자만 3등분해 Q1(하위)~Q3(상위)\n\n"
                            f"| 결과 | Ref n/ev | Q1 n/ev, HR | Q2 n/ev, HR | Q3 n/ev, HR | log-rank p |\n"
                            f"|---|---|---|---|---|---|\n")

        for ax, (ev_name, ecol, dcol) in zip(axes, EVENTS):
            sdf, ndrop = build_surv(d, ecol, dcol, lock)
            if sdf["e"].sum() < 3:
                ax.axis("off")
                continue
            s10 = sdf.copy(); s10["prob_poor"] = s10["prob_poor"] * 10
            fit_rows(task, ev_name, s10, ["prob_poor"], "univariable_per0.1", cox_rows)
            fit_rows(task, ev_name, s10, ["prob_poor", "age", "male"],
                      "adjusted_age_sex_per0.1", cox_rows)
            ho = s10[s10["split"] == "holdout_test"]
            if ho["e"].sum() >= 5:
                fit_rows(task, ev_name, ho, ["prob_poor"], "holdout_only_per0.1", cox_rows)

            km = km_panel(ax, sdf, thr, ev_name, fig_label)
            km["task"] = task; km["outcome"] = ev_name; km["threshold"] = thr
            km_rows.append(km)

            def cell(lab):
                if lab == "Ref(<thr)":
                    return f"{km[f'n_{lab}']}/{km[f'ev_{lab}']}"
                hr = km[f"HR_{lab}"]
                if np.isnan(hr):
                    return f"{km[f'n_{lab}']}/{km[f'ev_{lab}']}, n/a"
                return (f"{km[f'n_{lab}']}/{km[f'ev_{lab}']}, "
                        f"HR={hr:.2f}[{km[f'HR_lo_{lab}']:.2f},{km[f'HR_hi_{lab}']:.2f}] "
                        f"p={km[f'p_{lab}']:.1e}")

            md_sections.append(
                f"| {ev_name} | {cell('Ref(<thr)')} | {cell('Q1')} | {cell('Q2')} | "
                f"{cell('Q3')} | {km['logrank_p']:.1e} |\n")

        fig.tight_layout()
        png = f"{RESULT_DIR}/km_{task}.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        md_sections.append(f"\n![{task} KM](km_{task}.png)\n")
        print(f"saved {png}")

    cox_df = pd.DataFrame(cox_rows)
    cox_csv = f"{RESULT_DIR}/cox_results.csv"
    cox_df.to_csv(cox_csv, index=False)
    print(f"saved {cox_csv}  ({len(cox_df)} rows)")

    km_df = pd.DataFrame(km_rows)
    km_csv = f"{RESULT_DIR}/km_quartile_results.csv"
    km_df.to_csv(km_csv, index=False)
    print(f"saved {km_csv}  ({len(km_df)} rows)")

    md = ("# mRS 3개 모델 (퇴원/1년/3개월) x Death/Dementia/MACE — Cox + KM (4군)\n\n"
          f"BRAIN 내부 코호트만 사용(아산 외부검증 제외). Index=입원일, censor=데이터락 "
          f"{lock.date()}. 각 prob_poor는 해당 모델의 train-CV OOF/holdout 예측(자기 자신의 "
          "라벨을 학습에 쓰지 않음).\n\n"
          "**그룹 정의:** Ref = prob_poor < Youden 컷(train_cv_oof에서만 계산, honest). "
          "컷 이상인 환자만 따로 모아 3등분(tertile)해 Q1(그중 하위, 저위험)~Q3(상위, 최고위험). "
          "HR은 전부 Ref 대비.\n"
          + "".join(md_sections) +
          "\n\n## 파일\n\n"
          "- `cox_results.csv` — 연속형 Cox 계수(태스크x결과x모델x공변량, +0.1당 HR)\n"
          "- `km_quartile_results.csv` — Ref/Q1/Q2/Q3 4군 KM/Cox 요약\n"
          "- `km_<task>.png` — 태스크별 3패널(Death/Dementia/MACE) 4군 KM 그림\n"
          "- `cox_survival_3mrs.py` — 재현 코드\n")
    md_path = f"{RESULT_DIR}/SURVIVAL_COX_SUMMARY.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"saved {md_path}")


if __name__ == "__main__":
    main()
