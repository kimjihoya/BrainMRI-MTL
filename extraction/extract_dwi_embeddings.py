"""
DWI pooling study: does global average pooling (GAP) dilute small acute lesions
and kill the END1 signal?

Hypothesis
----
`_triad_encode` runs adaptive_avg_pool3d(1) on each Triad UNet encoder skip.
A small DWI lesion is diluted into the whole-brain mean -> signal lost in the frozen probe (END1 dwi=0.478).
Using max/std pooling instead of GAP may keep the local high activation (the lesion).

Method
----
Re-extract the DWI raw skips with 3 parameter-free poolings:
  - gap : adaptive_avg_pool3d  (the original baseline)
  - max : adaptive_max_pool3d  (preserves the local peak -> key test of the lesion hypothesis)
  - std : spatial standard deviation  (heterogeneity / lesion border)
Each is 1120-dim. gap+max concat (2240) is also tested.
tab/label/has_dwi are reused from the linear_probe_diagnostic cache (same extraction order).

Interpretation
----
  - if max/std/gap+max beat GAP (0.478) on END1 dwi with separated CIs
    -> pooling really was the bug; switch the model's pooling (imaging is alive).
  - if all sit near 0.5 -> END1 imaging truly has no signal in this data (simplification confirmed).

Usage:
    python pooling_reexperiment.py \
        --config configs/best/4task_single_film_t2frozen_triadpartial.yml \
        --gpu 2 --n_splits 5 --n_repeats 5 --pca_dim 50
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "shared")]  # repo root + shared/ modules
from linear_probe_diagnostic import (
    TARGET_TASKS, build_block, fit_predict, ci95,
)

warnings.filterwarnings("ignore", message="Mean of empty slice")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--gpu", type=str, default=None)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--pca_dim", type=int, default=50)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--base_cache", type=str, default="results/linear_probe_features.npz",
                   help="reuse tab/label/has_dwi (produced by linear_probe_diagnostic)")
    p.add_argument("--cache", type=str, default="results/pooling_variants_features.npz")
    p.add_argument("--use_cache", action="store_true")
    p.add_argument("--out_csv", type=str, default="results/pooling_variants.csv")
    return p.parse_args()


POOLINGS = ["gap", "max", "std", "gapmax", "topk2", "topk5", "topk10", "softmax", "maxmean"]


def _pool_skips(skips, mode):
    """skips: list of (B,C,D,H,W) → (B, sum_C) pooled feature."""
    b = skips[0].size(0)
    out = []
    for s in skips:
        if mode == "gap":
            out.append(F.adaptive_avg_pool3d(s, 1).view(b, -1))
        elif mode == "max":
            out.append(F.adaptive_max_pool3d(s, 1).view(b, -1))
        elif mode == "std":
            mean = F.adaptive_avg_pool3d(s, 1).view(b, -1)
            sq = F.adaptive_avg_pool3d(s * s, 1).view(b, -1)
            out.append(torch.sqrt(F.relu(sq - mean * mean) + 1e-6))
        elif mode == "gapmax":
            out.append(F.adaptive_avg_pool3d(s, 1).view(b, -1))
            out.append(F.adaptive_max_pool3d(s, 1).view(b, -1))
        elif mode.startswith("topk"):
            k = int(mode[4:])
            flat = s.flatten(2)                      # (B,C,DHW)
            kk = min(k, flat.size(2))
            out.append(flat.topk(kk, dim=2).values.mean(2).view(b, -1))  # focal mean
        elif mode == "softmax":
            flat = s.flatten(2)                      # (B,C,DHW)
            w = torch.softmax(flat, dim=2)           # self-attention by activation
            out.append((flat * w).sum(2).view(b, -1))
        elif mode == "maxmean":
            out.append(F.adaptive_max_pool3d(s, 1).view(b, -1))
            out.append(F.adaptive_avg_pool3d(s, 1).view(b, -1))
    return torch.cat(out, dim=1)


@torch.no_grad()
def extract_dwi_poolings(config, device):
    from model import MultiTaskModel
    from dataset import MultiTaskDataset, multitask_collate_fn, get_val_transform

    mcfg = config["model"]
    model = MultiTaskModel(
        t2_backbone="none",  # only DWI needed -> skip loading the T2 backbone
        dwi_backbone=mcfg.get("dwi_backbone", "triad"),
        triad_ckpt_path=config.get("triad", {}).get("ckpt_path"),
        task_names=TARGET_TASKS,
        dwi_feat_dim=mcfg.get("dwi_feat_dim", 256),
        tab_feat_dim=mcfg.get("tab_feat_dim", 128),
    ).to(device).eval()
    enc = model.dwi_enc.encoder

    image_size = tuple(config["data"]["size"])
    dwi_dir = config["data"].get("dwi_dir")
    clinical_csv = config["data"]["clinical_csv"]
    fin_csv = config["data"].get("outcomes_final_csv")

    def _extract(csv_path):
        ds = MultiTaskDataset(
            csv_path=csv_path, clinical_csv_path=clinical_csv,
            transform=get_val_transform(image_size), dwi_dir=dwi_dir,
            outcomes_final_csv=fin_csv,
        )
        loader = DataLoader(ds, batch_size=config["data"].get("batch_size", 16),
                            shuffle=False, num_workers=config["data"].get("num_workers", 4),
                            collate_fn=multitask_collate_fn)
        buf = {m: [] for m in POOLINGS}
        for _, dwi, _, _ in loader:
            dwi = dwi.to(device)
            skips = enc(dwi)
            for m in POOLINGS:
                buf[m].append(_pool_skips(skips, m).float().cpu().numpy())
        out = {m: np.concatenate(buf[m], axis=0) for m in POOLINGS}
        print(f"  {os.path.basename(csv_path)}: n={len(ds)}, "
              + ", ".join(f"{m}={out[m].shape[1]}" for m in POOLINGS))
        return out

    print("[extract] train DWI poolings ...")
    tr = _extract(config["data"]["csv_file"])
    print("[extract] holdout DWI poolings ...")
    te = _extract(config["data"]["val_csv"])
    return tr, te


def main():
    args = parse_args()
    from config_utils import load_config
    config = load_config(args.config)

    gpu = args.gpu or str(config.get("gpu", {}).get("visible_device", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- reuse tab/label/has_dwi from the existing cache (same extraction order) --
    base = np.load(args.base_cache, allow_pickle=True)
    tab_tr, tab_te = base["tr_tab"], base["te_tab"]
    has_dwi_tr, has_dwi_te = base["tr_has_dwi"], base["te_has_dwi"]
    labels_tr = {t: base[f"tr_label_{t}"] for t in TARGET_TASKS}
    labels_te = {t: base[f"te_label_{t}"] for t in TARGET_TASKS}

    # -- extract DWI pooling features or load cache --
    if args.use_cache and os.path.exists(args.cache):
        npz = np.load(args.cache, allow_pickle=True)
        dwi_tr = {m: npz[f"tr_{m}"] for m in POOLINGS}
        dwi_te = {m: npz[f"te_{m}"] for m in POOLINGS}
        print(f"[cache] loaded {args.cache}")
    else:
        dwi_tr, dwi_te = extract_dwi_poolings(config, device)
        save = {f"tr_{m}": v for m, v in dwi_tr.items()}
        save.update({f"te_{m}": v for m, v in dwi_te.items()})
        np.savez_compressed(args.cache, **save)
        print(f"[cache] saved {args.cache}")

    # sanity check: matching row counts
    n_tr = len(tab_tr)
    assert dwi_tr["gap"].shape[0] == n_tr, "train row count mismatch — regenerate the cache"

    rkf = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats,
                                  random_state=42)

    # feats dict: per-pooling DWI + gap baseline + tab
    def run_probe(task, mset, feats_tr, feats_te):
        y_tr, y_te = labels_tr[task], labels_te[task]
        mask = ~np.isnan(y_tr)
        if any(m != "tab" for m in mset):
            mask &= (has_dwi_tr == 1.0)
        idx = np.where(mask)[0]
        ysub = y_tr[idx]
        if len(np.unique(ysub)) < 2 or (ysub == 1).sum() < args.n_splits:
            return None

        aurocs, auprcs = [], []
        for tr_rel, ev_rel in rkf.split(idx, ysub):
            tr_i, ev_i = idx[tr_rel], idx[ev_rel]
            blk_tr, blk_ev = [], []
            for m in mset:
                X = feats_tr[m]
                btr, bev = build_block("dwi" if m != "tab" else "tab",
                                       X, tr_i, ev_i, args.pca_dim)
                blk_tr.append(btr); blk_ev.append(bev)
            Xtr = np.concatenate(blk_tr, 1); Xev = np.concatenate(blk_ev, 1)
            proba = fit_predict(Xtr, y_tr[tr_i], Xev, args.C)
            yev = y_tr[ev_i]
            if len(np.unique(yev)) < 2:
                continue
            aurocs.append(roc_auc_score(yev, proba))
            auprcs.append(average_precision_score(yev, proba))
        m_auc, lo, hi = ci95(aurocs)
        m_ap, _, _ = ci95(auprcs)
        return m_auc, lo, hi, m_ap, len(idx), int((ysub == 1).sum())

    # build per-modality feature lookup
    feats_tr = {**{m: dwi_tr[m] for m in POOLINGS}, "tab": tab_tr}
    feats_te = {**{m: dwi_te[m] for m in POOLINGS}, "tab": tab_te}

    modality_sets = []
    for m in POOLINGS:
        modality_sets.append([m])          # DWI pooling alone
        modality_sets.append([m, "tab"])   # dwi pooling + tab

    rows = []
    print("\n" + "=" * 72)
    print(f"DWI pooling study  |  CV={args.n_splits}x{args.n_repeats}  "
          f"PCA={args.pca_dim}  C={args.C}")
    print("(original GAP probe: END1 dwi=0.478, dwi+tab=0.534)")
    print("=" * 72)

    for task in TARGET_TASKS:
        print(f"\n### {task}  (train pos={int(np.nansum(labels_tr[task]==1))})")
        print(f"  {'modality':14s} {'CV AUROC [95% CI]':30s} {'CV AUPRC':10s} n")
        for mset in modality_sets:
            r = run_probe(task, mset, feats_tr, feats_te)
            if r is None:
                continue
            m_auc, lo, hi, m_ap, n, npos = r
            tag = "+".join(mset)
            star = "  *" if lo > 0.5 else ""
            print(f"  {tag:14s} {m_auc:.3f} [{lo:.3f},{hi:.3f}]{star:5s}"
                  f"        {m_ap:.3f}     {n}")
            rows.append({"task": task, "modality": tag, "cv_auroc": round(m_auc, 4),
                         "cv_lo": round(lo, 4), "cv_hi": round(hi, 4),
                         "cv_auprc": round(m_ap, 4), "n": n, "train_pos": npos})

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\nsaved results -> {args.out_csv}")
    print("Interpretation: '*' = CI above 0.5 (significant). If max/std/gapmax beat gap "
          "with separated CIs, pooling was the bug.")


if __name__ == "__main__":
    main()
