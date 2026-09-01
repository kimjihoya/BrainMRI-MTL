"""Modality / representation ablation for the 3 mRS poor-outcome targets
(dis_mrs, mrs1y, mrs3mo), reusing already-extracted FROZEN Triad/BrainIAC
embeddings (no GPU forward pass needed -> fast, runs entirely on CPU).

Mirrors the methodology used for this project's earlier deep-fusion reference
model: 5x5 repeated StratifiedKFold OOF, scaler/PCA fit on the TRAIN FOLD ONLY
(no leakage), LogisticRegression (class_weight="balanced"). Fixed pca_dim=50,
C=0.1 for all PCA'd branches (the representative config reported per pooling
type) so every row in the table is directly comparable -- this is a component
ablation, not a hyperparameter search.

Ablated components:
  clinical_only          - 27 admission tabular features only
  dwi_<pool>_only         - frozen DWI (Triad) embedding only, pool in {max,gap,std,gapmax}
  dwi_<pool>_fusion        - DWI(pool) + clinical
  t2_only                 - frozen T2 (BrainIAC) embedding only
  t2_fusion                - T2 + clinical
  trimodal_fusion          - DWI(gapmax) + T2 + clinical
  e2e_finetuned (reference) - NOT recomputed here; pulled from the already-trained
                               end-to-end fine-tune checkpoints (encoder weights
                               actually updated on this cohort) for direct comparison.

Output: results/ablation/ablation_results.csv (one row per target x modality)
Usage: python codes/mrs/analysis/ablation_modality_pooling.py
"""
import sys, os
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths

CLINICAL_CSV = paths.CLINICAL_CSV
LP_FEATS = "results/linear_probe_features.npz"
POOL_FEATS = "results/pooling_dwi_features.npz"
OUT_CSV = "results/ablation/ablation_results.csv"

TARGETS = {
    "dis_mrs": "dis_mrs",
    "mrs1y": "mrs1y",
    "mrs3mo": "mrs3mo",
}
# e2e reference numbers (already trained, encoder fine-tuned) for comparison only
E2E_REF = {
    "dis_mrs":  {"oof_auroc": 0.837, "ci": [0.803, 0.869], "holdout_auroc": 0.895, "n": 815, "n_pos": 279},
    "mrs1y":    {"oof_auroc": 0.835, "ci": [0.795, 0.873], "holdout_auroc": 0.790, "n": 615, "n_pos": 118},
    "mrs3mo":   {"oof_auroc": 0.810, "ci": [0.766, 0.850], "holdout_auroc": 0.765, "n": 686, "n_pos": 149},
}

PCA_DIM, C, N_SEEDS, N_FOLDS = 50, 0.1, 5, 5
rng = np.random.default_rng(0)


def load_features():
    lp = np.load(LP_FEATS, allow_pickle=True)
    pp = np.load(POOL_FEATS, allow_pickle=True)
    pat = np.concatenate([lp["tr_pat_id"], lp["te_pat_id"]]).astype(str)
    tab = np.concatenate([lp["tr_tab"], lp["te_tab"]], 0)
    t2 = np.concatenate([lp["tr_t2"], lp["te_t2"]], 0)
    pools = {k: np.concatenate([pp[f"tr_{k}"], pp[f"te_{k}"]], 0) for k in ["max", "gap", "std", "gapmax"]}
    return pat, tab, t2, pools


def load_labels():
    clin = pd.read_csv(CLINICAL_CSV, dtype={"record_id": str}, encoding="utf-8-sig")
    clin = clin.set_index("record_id")
    labels = {}
    for name, col in TARGETS.items():
        v = pd.to_numeric(clin[col], errors="coerce")
        v = v.where(v != 9)  # 9 = unknown, exclude
        yl = pd.Series(np.nan, index=clin.index)
        yl[v >= 3] = 1.0
        yl[v <= 2] = 0.0
        labels[name] = yl
    return labels


def block(Xfull, ti, ei, pca_dim, do_pca):
    Xtr, Xev = Xfull[ti], Xfull[ei]
    cm = np.nanmean(Xtr, 0); cm = np.where(np.isnan(cm), 0, cm)
    Xtr = np.where(np.isnan(Xtr), cm, Xtr); Xev = np.where(np.isnan(Xev), cm, Xev)
    sc = StandardScaler().fit(Xtr); Xtr, Xev = sc.transform(Xtr), sc.transform(Xev)
    if do_pca and Xtr.shape[1] > pca_dim:
        pc = PCA(pca_dim, random_state=0).fit(Xtr); Xtr, Xev = pc.transform(Xtr), pc.transform(Xev)
    return Xtr, Xev


def bootstrap_ci(y, p, n=2000):
    a, idx = [], np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), True)
        if y[b].sum() < 2 or (1 - y[b]).sum() < 2:
            continue
        a.append(roc_auc_score(y[b], p[b]))
    return np.percentile(a, [2.5, 97.5])


def oof_auc(y, blocks, pca_flags):
    """blocks: list of feature matrices to concat; pca_flags: parallel list of bool (apply PCA)."""
    n = len(y)
    o = np.zeros(n); c = np.zeros(n); ii = np.arange(n)
    for seed in range(N_SEEDS):
        for tr, ev in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(ii, y):
            Xtr_parts, Xev_parts = [], []
            for Xb, do_pca in zip(blocks, pca_flags):
                bt, be = block(Xb, tr, ev, PCA_DIM, do_pca)
                Xtr_parts.append(bt); Xev_parts.append(be)
            Xtr = np.concatenate(Xtr_parts, 1); Xev = np.concatenate(Xev_parts, 1)
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
            clf.fit(Xtr, y[tr].astype(int))
            o[ev] += clf.predict_proba(Xev)[:, 1]; c[ev] += 1
    return o / np.maximum(c, 1)


def run_target(name, y_full, pat, tab, t2, pools, rows):
    mask = ~np.isnan(y_full)
    idx = np.where(mask)[0]
    y = y_full[idx].astype(int)
    print(f"\n=== {name}  n={len(y)}  pos={int(y.sum())} ({y.mean():.1%}) ===", flush=True)

    configs = [("clinical_only", [tab[idx]], [False])]
    for pool in ["max", "gap", "std", "gapmax"]:
        configs.append((f"dwi_{pool}_only", [pools[pool][idx]], [True]))
        configs.append((f"dwi_{pool}_fusion", [pools[pool][idx], tab[idx]], [True, False]))
    configs.append(("t2_only", [t2[idx]], [True]))
    configs.append(("t2_fusion", [t2[idx], tab[idx]], [True, False]))
    configs.append(("trimodal_fusion_dwi_gapmax_t2", [pools["gapmax"][idx], t2[idx], tab[idx]], [True, True, False]))

    for label, blocks, pca_flags in configs:
        t0 = __import__("time").time()
        p = oof_auc(y, blocks, pca_flags)
        auc = roc_auc_score(y, p)
        lo, hi = bootstrap_ci(y, p)
        ap = average_precision_score(y, p)
        dt = __import__("time").time() - t0
        print(f"  {label:32s} AUROC={auc:.3f} [{lo:.3f},{hi:.3f}]  AP={ap:.3f}  ({dt:.1f}s)", flush=True)
        rows.append({"target": name, "modality": label, "n": len(y), "n_pos": int(y.sum()),
                      "eval": "5x5_OOF_frozen_logreg", "auroc": round(auc, 4),
                      "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "ap": round(ap, 4),
                      "pca_dim": PCA_DIM if any(pca_flags) else None, "C": C})

    ref = E2E_REF[name]
    rows.append({"target": name, "modality": "e2e_finetuned_REFERENCE", "n": ref["n"], "n_pos": ref["n_pos"],
                  "eval": "tr_only_CV_5x5x3seed (encoder fine-tuned, NOT this script)",
                  "auroc": ref["oof_auroc"], "ci_lo": ref["ci"][0], "ci_hi": ref["ci"][1],
                  "ap": None, "pca_dim": None, "C": None})
    print(f"  {'e2e_finetuned (reference)':32s} tr-CV AUROC={ref['oof_auroc']:.3f}  "
          f"holdout AUROC={ref['holdout_auroc']:.3f}")


def main():
    pat, tab, t2, pools = load_features()
    labels = load_labels()
    # align labels to the pat order used by the cached feature arrays
    y_by_target = {}
    for name, yl in labels.items():
        y_by_target[name] = np.array([yl[p] if p in yl.index else np.nan for p in pat])

    rows = []
    for name in TARGETS:
        run_target(name, y_by_target[name], pat, tab, t2, pools, rows)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
