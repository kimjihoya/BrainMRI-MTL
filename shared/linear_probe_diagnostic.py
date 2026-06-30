"""
Linear-probe diagnostic: do frozen pretrained backbone features carry signal for each task?

Purpose
----
The end-to-end MTL model trains two 3D backbones (BrainIAC ViT, Triad UNet) + a fusion stack
on ~650 patients / a few dozen positives -> large CV->holdout gap (suspected overfitting).

This script measures the opposite extreme:
  - freeze the backbones, extract only the **untrained raw pooled features** (no outcome leakage)
  - a **LogisticRegression linear probe** per modality (tab / t2 / dwi / combinations)
  - **RepeatedStratifiedKFold (n_splits×n_repeats)** + 95% CI + AUPRC
  - preprocessing (scaler/PCA) is **fit on the fold train only** (no leakage)
  - patients without DWI are auto-excluded from dwi-containing sets

Interpretation
----
  - if a modality's frozen probe cannot beat random (0.5) meaningfully
    -> the end-to-end model is memorizing, not learning signal from that modality.
  - if probe CV is similar to the MTL CV -> the heavy fusion stack adds nothing.
  - if probe holdout is more stable than the MTL holdout -> shrinking capacity is the answer.

Usage:
    python linear_probe_diagnostic.py \
        --config configs/best/4task_single_film_t2frozen_triadpartial.yml \
        --gpu 1 --n_splits 5 --n_repeats 5 --pca_dim 50

    # reuse the extraction cache (backbone inference is slow, do it once)
    python linear_probe_diagnostic.py --config ... --use_cache
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--gpu", type=str, default=None,
                   help="CUDA device index (default: config gpu.visible_device)")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--pca_dim", type=int, default=50,
                   help="image-feature PCA dim (0=off). tab is not PCA-reduced")
    p.add_argument("--C", type=float, default=1.0, help="LogisticRegression L2 strength")
    p.add_argument("--cache", type=str, default="results/linear_probe_features.npz")
    p.add_argument("--use_cache", action="store_true",
                   help="skip extraction and use the cache")
    p.add_argument("--out_csv", type=str, default="results/linear_probe_diagnostic.csv")
    return p.parse_args()


# the 4 target tasks to evaluate
TARGET_TASKS = ["end1_binary", "mortality_binary", "dementia_binary", "mace_binary"]


# ---------------------------------------------------------------------------
# Feature extraction (frozen pretrained backbone, no trained projection)
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_raw_features(config, device):
    """Extract raw pretrained pooled features: t2(768/1120), dwi(1120), tab(raw)."""
    from model import MultiTaskModel, _triad_encode, _brainmvp_encode
    from dataset import (
        MultiTaskDataset, multitask_collate_fn, get_val_transform,
        ALL_FEATURES,
    )

    # -- load only the pretrained backbones (ignore fusion/head) --
    mcfg = config["model"]
    model = MultiTaskModel(
        t2_backbone=mcfg.get("t2_backbone", "brainiac"),
        dwi_backbone=mcfg.get("dwi_backbone", "triad"),
        ckpt_path=config.get("simclrvit", {}).get("ckpt_path"),
        triad_ckpt_path=config.get("triad", {}).get("ckpt_path"),
        brainmvp_ckpt_path=config.get("brainmvp", {}).get("ckpt_path"),
        task_names=TARGET_TASKS,
        image_feat_dim=mcfg.get("image_feat_dim", 128),
        dwi_feat_dim=mcfg.get("dwi_feat_dim", 256),
        tab_feat_dim=mcfg.get("tab_feat_dim", 128),
        film=mcfg.get("film", False),
    ).to(device).eval()

    t2_bb = model.t2_backbone
    has_t2 = t2_bb != "none"
    has_dwi_bb = model.dwi_backbone != "none"

    image_size = tuple(config["data"]["size"])
    dwi_dir = config["data"].get("dwi_dir")
    clinical_csv = config["data"]["clinical_csv"]
    fin_csv = config["data"].get("outcomes_final_csv")

    def _raw_t2(imgs):
        if t2_bb == "brainiac":
            return model.backbone(imgs)                       # (B, 768)
        if t2_bb == "triad":
            return _triad_encode(model.backbone, imgs)        # (B, 1120)
        if t2_bb == "brainmvp":
            return _brainmvp_encode(model.backbone, imgs)     # (B, 512)
        return None

    def _extract_split(csv_path):
        ds = MultiTaskDataset(
            csv_path=csv_path, clinical_csv_path=clinical_csv,
            transform=get_val_transform(image_size), dwi_dir=dwi_dir,
            outcomes_final_csv=fin_csv,
        )
        loader = DataLoader(ds, batch_size=config["data"].get("batch_size", 16),
                            shuffle=False, num_workers=config["data"].get("num_workers", 4),
                            collate_fn=multitask_collate_fn)
        t2_list, dwi_list = [], []
        for imgs, dwi, tab, lbls in loader:
            imgs, dwi = imgs.to(device), dwi.to(device)
            if has_t2:
                t2_list.append(_raw_t2(imgs).float().cpu().numpy())
            if has_dwi_bb:
                dwi_list.append(_triad_encode(model.dwi_enc, dwi).float().cpu().numpy())

        df = ds.df.reset_index(drop=True)
        pat_id = df["pat_id"].astype(str).values

        # has_dwi: whether the DWI file actually exists
        has_dwi = np.array([
            1.0 if (dwi_dir and os.path.exists(os.path.join(dwi_dir, f"{p}.nii.gz")))
            else 0.0 for p in pat_id
        ])

        # raw tabular (straight from df; standardization happens inside the CV fold)
        tab_raw = np.column_stack([
            pd.to_numeric(df[c], errors="coerce").values if c in df.columns
            else np.full(len(df), np.nan)
            for c in ALL_FEATURES
        ]).astype(float)

        # labels (4 target)
        labels = {
            t: pd.to_numeric(df[t], errors="coerce").values.astype(float)
            if t in df.columns else np.full(len(df), np.nan)
            for t in TARGET_TASKS
        }

        out = {
            "tab": tab_raw, "has_dwi": has_dwi, "pat_id": pat_id,
            **{f"label_{t}": labels[t] for t in TARGET_TASKS},
        }
        if has_t2:
            out["t2"] = np.concatenate(t2_list, axis=0)
        if has_dwi_bb:
            out["dwi"] = np.concatenate(dwi_list, axis=0)
        print(f"  extracted {os.path.basename(csv_path)}: n={len(df)}, "
              f"t2={out.get('t2', np.zeros((0,0))).shape}, "
              f"dwi={out.get('dwi', np.zeros((0,0))).shape}, tab={tab_raw.shape}")
        return out

    print("[extract] train split ...")
    train = _extract_split(config["data"]["csv_file"])
    print("[extract] holdout split ...")
    test = _extract_split(config["data"]["val_csv"])
    return train, test


# ---------------------------------------------------------------------------
# Feature assembly (per block: impute -> scale -> PCA); fit on train indices only
# ---------------------------------------------------------------------------
def build_block(name, X_full, train_idx, eval_idx, pca_dim):
    """Single-modality block: fit scaler/PCA on train_idx -> transform train+eval."""
    Xtr, Xev = X_full[train_idx], X_full[eval_idx]
    # mean-impute (train mean)
    col_mean = np.nanmean(Xtr, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    Xtr = np.where(np.isnan(Xtr), col_mean, Xtr)
    Xev = np.where(np.isnan(Xev), col_mean, Xev)
    # standardize
    sc = StandardScaler().fit(Xtr)
    Xtr, Xev = sc.transform(Xtr), sc.transform(Xev)
    # PCA (image blocks only, not tab)
    if pca_dim and name in ("t2", "dwi") and Xtr.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=0).fit(Xtr)
        Xtr, Xev = pca.transform(Xtr), pca.transform(Xev)
    return Xtr, Xev


def assemble(modality_set, feats, train_idx, eval_idx, pca_dim):
    blocks_tr, blocks_ev = [], []
    for m in modality_set:
        if m not in feats:
            continue
        btr, bev = build_block(m, feats[m], train_idx, eval_idx, pca_dim)
        blocks_tr.append(btr)
        blocks_ev.append(bev)
    return np.concatenate(blocks_tr, axis=1), np.concatenate(blocks_ev, axis=1)


# ---------------------------------------------------------------------------
# Probe evaluation
# ---------------------------------------------------------------------------
def fit_predict(Xtr, ytr, Xev, C):
    clf = LogisticRegression(
        C=C, class_weight="balanced", max_iter=2000, solver="lbfgs",
    )
    clf.fit(Xtr, ytr.astype(int))
    return clf.predict_proba(Xev)[:, 1]


def ci95(vals):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    m = vals.mean()
    sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return m, m - 1.96 * sem, m + 1.96 * sem


def main():
    args = parse_args()
    from config_utils import load_config
    config = load_config(args.config)

    gpu = args.gpu or str(config.get("gpu", {}).get("visible_device", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- extract features or load cache --
    if args.use_cache and os.path.exists(args.cache):
        print(f"[cache] loaded {args.cache}")
        npz = np.load(args.cache, allow_pickle=True)
        train = {k[2:]: npz[k] for k in npz.files if k.startswith("tr_")}
        test = {k[2:]: npz[k] for k in npz.files if k.startswith("te_")}
    else:
        train, test = extract_raw_features(config, device)
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        save = {f"tr_{k}": v for k, v in train.items()}
        save.update({f"te_{k}": v for k, v in test.items()})
        np.savez_compressed(args.cache, **save)
        print(f"[cache] saved {args.cache}")

    avail = [m for m in ("t2", "dwi", "tab") if m in train]
    modality_sets = [[m] for m in avail]
    if "t2" in avail and "dwi" in avail:
        modality_sets.append(["t2", "dwi"])
    for img in ("t2", "dwi"):
        if img in avail:
            modality_sets.append([img, "tab"])
    if len(avail) >= 2:
        modality_sets.append(avail)  # all

    rkf = RepeatedStratifiedKFold(
        n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=42,
    )

    rows = []
    print("\n" + "=" * 78)
    print(f"Linear-probe diagnostic  |  CV={args.n_splits}x{args.n_repeats}  "
          f"PCA={args.pca_dim}  C={args.C}")
    print("=" * 78)

    for task in TARGET_TASKS:
        y_tr_full = train[f"label_{task}"]
        y_te_full = test[f"label_{task}"]
        print(f"\n### {task}  (train pos={int(np.nansum(y_tr_full==1))}, "
              f"holdout pos={int(np.nansum(y_te_full==1))})")
        print(f"  {'modality':14s} {'CV AUROC [95% CI]':28s} "
              f"{'CV AUPRC':16s} {'holdout AUROC/AUPRC':22s} n_used")

        for mset in modality_sets:
            # usable-sample mask: label present + (has_dwi==1 if dwi is included)
            mask = ~np.isnan(y_tr_full)
            if "dwi" in mset:
                mask &= (train["has_dwi"] == 1.0)
            idx_all = np.where(mask)[0]
            y_sub = y_tr_full[idx_all]
            if len(np.unique(y_sub)) < 2 or (y_sub == 1).sum() < args.n_splits:
                continue

            # ── CV ──
            aurocs, auprcs = [], []
            for tr_rel, ev_rel in rkf.split(idx_all, y_sub):
                tr_idx, ev_idx = idx_all[tr_rel], idx_all[ev_rel]
                Xtr, Xev = assemble(mset, train, tr_idx, ev_idx, args.pca_dim)
                proba = fit_predict(Xtr, y_tr_full[tr_idx], Xev, args.C)
                yev = y_tr_full[ev_idx]
                if len(np.unique(yev)) < 2:
                    continue
                aurocs.append(roc_auc_score(yev, proba))
                auprcs.append(average_precision_score(yev, proba))
            m_auc, lo, hi = ci95(aurocs)
            m_ap, _, _ = ci95(auprcs)

            # -- holdout (fit on all train -> predict holdout) --
            te_mask = ~np.isnan(y_te_full)
            if "dwi" in mset:
                te_mask &= (test["has_dwi"] == 1.0)
            te_idx = np.where(te_mask)[0]
            ho_auc = ho_ap = np.nan
            if len(te_idx) > 0 and len(np.unique(y_te_full[te_idx])) >= 2:
                # fit on train (idx_all), predict holdout (te_idx) — per-block train fit
                Xtr, Xho = assemble_train_test(mset, train, test, idx_all, te_idx, args.pca_dim)
                proba_ho = fit_predict(Xtr, y_tr_full[idx_all], Xho, args.C)
                ho_auc = roc_auc_score(y_te_full[te_idx], proba_ho)
                ho_ap = average_precision_score(y_te_full[te_idx], proba_ho)

            tag = "+".join(mset)
            print(f"  {tag:14s} {m_auc:.3f} [{lo:.3f},{hi:.3f}]      "
                  f"{m_ap:.3f}            {ho_auc:.3f}/{ho_ap:.3f}        {len(idx_all)}")
            rows.append({
                "task": task, "modality": tag,
                "cv_auroc": round(m_auc, 4), "cv_auroc_lo": round(lo, 4),
                "cv_auroc_hi": round(hi, 4), "cv_auprc": round(m_ap, 4),
                "holdout_auroc": round(ho_auc, 4), "holdout_auprc": round(ho_ap, 4),
                "n_train": len(idx_all), "n_holdout": len(te_idx),
                "train_pos": int((y_sub == 1).sum()),
            })

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\nsaved results -> {args.out_csv}")


def assemble_train_test(modality_set, train, test, train_idx, test_idx, pca_dim):
    """For holdout eval: fit each train block on train_idx, transform the test from separate feats."""
    blocks_tr, blocks_te = [], []
    for m in modality_set:
        if m not in train or m not in test:
            continue
        Xtr_raw = train[m][train_idx]
        Xte_raw = test[m][test_idx]
        col_mean = np.nanmean(Xtr_raw, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        Xtr_raw = np.where(np.isnan(Xtr_raw), col_mean, Xtr_raw)
        Xte_raw = np.where(np.isnan(Xte_raw), col_mean, Xte_raw)
        sc = StandardScaler().fit(Xtr_raw)
        Xtr_raw, Xte_raw = sc.transform(Xtr_raw), sc.transform(Xte_raw)
        if pca_dim and m in ("t2", "dwi") and Xtr_raw.shape[1] > pca_dim:
            pca = PCA(n_components=pca_dim, random_state=0).fit(Xtr_raw)
            Xtr_raw, Xte_raw = pca.transform(Xtr_raw), pca.transform(Xte_raw)
        blocks_tr.append(Xtr_raw)
        blocks_te.append(Xte_raw)
    return np.concatenate(blocks_tr, axis=1), np.concatenate(blocks_te, axis=1)


if __name__ == "__main__":
    main()
