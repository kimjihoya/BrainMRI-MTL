"""Hyperparameter (C, PCA-dim) robustness sweep for the frozen modality ablation.

Answers: is the "imaging only helps at 3mo, not discharge/1y" finding
(ablation_modality_pooling.py) an artifact of the fixed C=0.1/PCA=50 choice,
or does it hold across the LogisticRegression regularization / PCA-dimension
grid? Only the 3 decision-relevant configs per target are swept (clinical_only,
best DWI-only pool, best DWI+clinical fusion pool -- pools picked from the
prior ablation's per-target winners), not the full 13-config x 3-target board,
to keep this tractable on CPU.

Same protocol as ablation_modality_pooling.py (5x5 OOF, scaler/PCA fit on
train fold only). Run from repo root: python codes/mrs/analysis/ablation_c_pca_sweep.py
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_modality_pooling import load_features, load_labels, oof_auc, bootstrap_ci
from sklearn.metrics import roc_auc_score, average_precision_score

OUT_CSV = "results/ablation/ablation_c_pca_sweep_results.csv"

# per-target best pool from the prior modality ablation
BEST_POOL = {"dis_mrs": "max", "mrs1y": "max", "mrs3mo": "gapmax"}
DWI_ONLY_BEST_POOL = {"dis_mrs": "gap", "mrs1y": "max", "mrs3mo": "gap"}

C_GRID = [0.03, 0.1, 0.3, 1.0]
PCA_GRID = [20, 50, 80]


def main():
    pat, tab, t2, pools = load_features()
    labels = load_labels()
    rows = []

    for target in ["dis_mrs", "mrs1y", "mrs3mo"]:
        yl = labels[target]
        y_full = np.array([yl[p] if p in yl.index else np.nan for p in pat])
        mask = ~np.isnan(y_full); idx = np.where(mask)[0]; y = y_full[idx].astype(int)
        print(f"\n=== {target}  n={len(y)} pos={int(y.sum())} ===", flush=True)

        # clinical_only: C sweep only (no PCA branch)
        for C in C_GRID:
            import ablation_modality_pooling as amp
            amp.C = C
            p = oof_auc(y, [tab[idx]], [False])
            auc = roc_auc_score(y, p); lo, hi = bootstrap_ci(y, p)
            print(f"  clinical_only        C={C:<5} AUROC={auc:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
            rows.append({"target": target, "modality": "clinical_only", "C": C, "pca_dim": None,
                          "auroc": round(auc, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                          "ap": round(average_precision_score(y, p), 4), "n": len(y), "n_pos": int(y.sum())})

        dwi_pool = DWI_ONLY_BEST_POOL[target]
        fus_pool = BEST_POOL[target]
        for modality, blocks_fn, pool_name in [
            ("dwi_only", lambda pd_: [pools[dwi_pool][idx]], dwi_pool),
            ("dwi_fusion", lambda pd_: [pools[fus_pool][idx], tab[idx]], fus_pool),
        ]:
            pca_flags = [True] if modality == "dwi_only" else [True, False]
            for pd_ in PCA_GRID:
                for C in C_GRID:
                    import ablation_modality_pooling as amp
                    amp.C = C; amp.PCA_DIM = pd_
                    p = oof_auc(y, blocks_fn(pd_), pca_flags)
                    auc = roc_auc_score(y, p); lo, hi = bootstrap_ci(y, p)
                    print(f"  {modality}({pool_name})  pca={pd_:<3} C={C:<5} AUROC={auc:.3f} "
                          f"[{lo:.3f},{hi:.3f}]", flush=True)
                    rows.append({"target": target, "modality": f"{modality}_{pool_name}", "C": C,
                                  "pca_dim": pd_, "auroc": round(auc, 4), "ci_lo": round(lo, 4),
                                  "ci_hi": round(hi, 4), "ap": round(average_precision_score(y, p), 4),
                                  "n": len(y), "n_pos": int(y.sum())})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV}  ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
