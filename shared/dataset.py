import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized,
    NormalizeIntensityd, RandAffined, RandFlipd,
    RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd, ToTensord,
)

BINARY_TASKS = ["end1_binary", "mortality_binary", "dementia_binary", "mace_binary",
                "mace_early_binary", "mace_lm_binary"]

REGRESSION_TASKS = ["dis_mrs", "mrs_3mo_ord", "mrs_1y_ord"]
REG_NORMALIZERS  = {"dis_mrs": 6.0, "mrs_3mo_ord": 6.0, "mrs_1y_ord": 6.0}

TASK_NAMES = BINARY_TASKS + REGRESSION_TASKS

# Continuous features — z-score normalised, mean-imputed
CONT_FEATURES = [
    "age", "sbp", "dbp", "ini_nih", "pre_mrs",
    "wbc", "hb", "plt", "pt",
    "cr", "crp", "fbs", "i_glu", "ha1c",
    "hdl", "ldl",
    "bmi",
]  # 17

# Binary features — kept as-is (0/1), mean-imputed
BIN_FEATURES = [
    "male",
    "hx_htn", "hx_dm", "hx_af", "hx_hl", "hx_str", "hx_chd", "smok",
    "ekg_af",
]  # 9

# -- FLAIR-DWI mismatch scalars (opt-in: env USE_FLAIR_MISMATCH=1) --
# The lever that gave the END1 frozen probe 0.648. Label-independent per-volume scalars.
USE_FLAIR_MISMATCH = os.environ.get("USE_FLAIR_MISMATCH", "0") == "1"
_MM_PATH = os.environ.get("FLAIR_MISMATCH_NPZ",
                          os.path.join(os.path.dirname(__file__), "results/flair_mismatch_feats.npz"))
MISMATCH_FEATS, MISMATCH_NAMES, MISMATCH_DIM = {}, [], 0
if USE_FLAIR_MISMATCH and os.path.exists(_MM_PATH):
    _mm = np.load(_MM_PATH, allow_pickle=True)
    MISMATCH_NAMES = [str(s) for s in _mm["names"]]
    MISMATCH_DIM = len(MISMATCH_NAMES)
    MISMATCH_FEATS = {str(p): v.astype(np.float32)
                      for p, v in zip(_mm["pat_id"].astype(str), _mm["feats"])}
    print(f"[dataset] FLAIR mismatch ON: {MISMATCH_DIM} feats, {len(MISMATCH_FEATS)} patients")

TAB_FEATURE_DIM = len(CONT_FEATURES) + len(BIN_FEATURES) + 1 + MISMATCH_DIM  # 26 + has_dwi (+22 mismatch)

ALL_FEATURES = CONT_FEATURES + BIN_FEATURES + ["has_dwi"] + MISMATCH_NAMES


def _load_clinical(clinical_csv_path: str) -> pd.DataFrame:
    clin = pd.read_csv(clinical_csv_path, dtype={"record_id": str}, encoding="utf-8-sig")
    clin = clin.rename(columns={"record_id": "pat_id"})

    # Derived feature
    clin["bmi"] = clin["wt"] / (clin["ht"] / 100) ** 2

    # Encode sex: m→1, f→0
    clin["male"] = (clin["male"] == "m").astype(float)

    # Cast all feature columns to numeric
    for col in CONT_FEATURES + BIN_FEATURES:
        if col in clin.columns and col != "bmi":
            clin[col] = pd.to_numeric(clin[col], errors="coerce")

    # ── New outcome labels ──────────────────────────────────────────────────
    # end1_binary comes from outcomes_final.csv (corrected v2: 95 pos / 893 neg, fully labelled).
    #   The old all_data end1/no_end was a labelling error and is discarded.
    _oc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/csvs/outcomes_final.csv")
    _oc = pd.read_csv(_oc_path, dtype={"record_id": str}, encoding="utf-8-sig")
    _oc.columns = [c.strip().replace("﻿", "") for c in _oc.columns]
    _end1_map = dict(zip(_oc["record_id"].astype(str),
                         pd.to_numeric(_oc["end1"], errors="coerce")))
    clin["end1_binary"] = clin["pat_id"].astype(str).map(_end1_map).astype(float)

    # mortality1y_binary: (dis_st==1 OR mrs1y==6) → 1; mrs1y 0-5 → 0; else NaN
    dis_st   = pd.to_numeric(clin["dis_st"],  errors="coerce")
    mrs1y_r  = pd.to_numeric(clin["mrs1y"],   errors="coerce")
    died     = (dis_st == 1) | (mrs1y_r == 6)
    survived = ~died & mrs1y_r.between(0, 5)
    clin["mortality1y_binary"] = float("nan")
    clin.loc[died,     "mortality1y_binary"] = 1.0
    clin.loc[survived, "mortality1y_binary"] = 0.0

    return clin


_IMG_KEYS = ["image", "dwi"]


def get_train_transform(image_size=(96, 96, 96)):
    return Compose([
        LoadImaged(keys=_IMG_KEYS),
        EnsureChannelFirstd(keys=_IMG_KEYS),
        Resized(keys=_IMG_KEYS, spatial_size=image_size, mode="trilinear"),
        NormalizeIntensityd(keys=_IMG_KEYS, nonzero=True, channel_wise=True),
        RandAffined(
            keys=_IMG_KEYS,
            rotate_range=(0.1, 0.1, 0.1),
            translate_range=(5, 5, 5),
            scale_range=(0.1, 0.1, 0.1),
            prob=0.5,
            padding_mode="border",
        ),
        RandFlipd(keys=_IMG_KEYS, spatial_axis=[2], prob=0.5),
        RandGaussianSmoothd(keys=_IMG_KEYS, prob=0.2),
        RandGaussianNoised(keys=_IMG_KEYS, prob=0.2, std=0.05),
        RandAdjustContrastd(keys=_IMG_KEYS, prob=0.2, gamma=(0.7, 1.3)),
        ToTensord(keys=_IMG_KEYS),
    ])


def get_val_transform(image_size=(96, 96, 96)):
    return Compose([
        LoadImaged(keys=_IMG_KEYS),
        EnsureChannelFirstd(keys=_IMG_KEYS),
        Resized(keys=_IMG_KEYS, spatial_size=image_size, mode="trilinear"),
        NormalizeIntensityd(keys=_IMG_KEYS, nonzero=True, channel_wise=True),
        ToTensord(keys=_IMG_KEYS),
    ])


class MultiTaskDataset(Dataset):
    """
    Required columns in csv_path:
        pat_id, t2_path, mrs3mo_binary, mrs1y_binary

    clinical_csv_path (lab_outcome.csv) supplies:
        tabular features, end1_binary, mortality1y_binary

    dwi_dir: directory containing {pat_id}.nii.gz DWI files.
             If None, DWI branch is disabled.

    feature_stats: pass train-set stats to val/test dataset so normalisation
                   is consistent. If None, stats are computed from this split.
    """

    def __init__(self, csv_path: str, clinical_csv_path: str,
                 transform=None, feature_stats=None, dwi_dir: str = None,
                 extended_outcomes_csv: str = None,
                 outcomes_final_csv: str = None):
        self.dwi_dir = dwi_dir
        main  = pd.read_csv(csv_path, dtype={"pat_id": str})
        clin  = _load_clinical(clinical_csv_path)

        # Only pull outcome cols from clin if they're not already in the main CSV
        new_outcome_cols = [c for c in ["end1_binary", "mortality1y_binary"]
                            if c not in main.columns]
        clin_cols = (["pat_id"] + new_outcome_cols
                     + [c for c in CONT_FEATURES + BIN_FEATURES if c in clin.columns])
        self.df = main.merge(clin[clin_cols], on="pat_id", how="left")

        # -- outcomes_final_csv: replace all outcomes in one place --
        if outcomes_final_csv:
            fin = pd.read_csv(outcomes_final_csv)
            fin.columns = [c.strip().replace("﻿", "") for c in fin.columns]
            fin = fin.rename(columns={
                "record_id": "pat_id",
                "Dementia":  "dementia_binary",
                "MACE":      "mace_binary",
                "Death":     "mortality_binary",
                "end1":      "end1_binary",   # corrected v2 column (end1, 95 pos). Old END (61) is discarded.
            })
            fin["pat_id"] = fin["pat_id"].astype(str)

            # binary mrs: 9 (lost to follow-up) -> NaN, 0-2 -> 0, 3-6 -> 1
            for raw_col, bin_col in [("mrs_3mo", "mrs3mo_binary"), ("mrs_1y", "mrs1y_binary")]:
                if raw_col in fin.columns:
                    raw = pd.to_numeric(fin[raw_col], errors="coerce")
                    fin[bin_col] = float("nan")
                    fin.loc[raw.between(0, 2), bin_col] = 0.0
                    fin.loc[raw.between(3, 6), bin_col] = 1.0
                    # 9 -> NaN (lost to follow-up, already NaN)

            # ordinal columns: 9 -> NaN, normalization happens in __getitem__
            for raw_col, ord_col in [("mrs_3mo", "mrs_3mo_ord"), ("mrs_1y", "mrs_1y_ord")]:
                if raw_col in fin.columns:
                    raw = pd.to_numeric(fin[raw_col], errors="coerce")
                    fin[ord_col] = raw.where(raw.between(0, 6), other=float("nan"))

            # dis_mrs, dis_nih used directly (NaN kept as-is)
            for col in ["dis_mrs", "dis_nih"]:
                if col in fin.columns:
                    fin[col] = pd.to_numeric(fin[col], errors="coerce")

            outcome_cols = [c for c in BINARY_TASKS + REGRESSION_TASKS if c in fin.columns]
            self.df = self.df.drop(columns=[c for c in outcome_cols if c in self.df.columns])
            self.df = self.df.merge(fin[["pat_id"] + outcome_cols], on="pat_id", how="left")

            # binary outcome: blank = negative (0). Every cohort patient exists in outcomes,
            # so NaN = 'no event' -> set explicit 0 (avoids the bug where masking erases negatives).
            # mrs/regression: NaN = lost to follow-up, so never fill them.
            for c in BINARY_TASKS:
                if c in self.df.columns:
                    self.df[c] = pd.to_numeric(self.df[c], errors="coerce").fillna(0.0)
        else:
            # ── Extended outcomes: mortality_binary, dementia_binary, mace_binary ──
            if extended_outcomes_csv:
                ext = pd.read_csv(extended_outcomes_csv, dtype={"pat_id": str})
            else:
                import os
                default_ext = os.path.join(os.path.dirname(clinical_csv_path),
                                           "outcomes_extended.csv")
                ext = pd.read_csv(default_ext, dtype={"pat_id": str}) \
                    if os.path.exists(default_ext) else None

            if ext is not None:
                for col in ["mortality_binary", "dementia_binary", "mace_binary"]:
                    if col in ext.columns and col not in self.df.columns:
                        self.df = self.df.merge(ext[["pat_id", col]], on="pat_id", how="left")

        self.transform = transform
        self.feature_stats = feature_stats if feature_stats is not None else self._compute_feature_stats()

    def compute_pos_weights(self, max_weight: float = 20.0) -> dict:
        """neg/pos ratio per task, capped at max_weight. Regression tasks get 1.0."""
        weights = {}
        for task in TASK_NAMES:
            if task in REGRESSION_TASKS:
                weights[task] = 1.0
                continue
            if task not in self.df.columns:   # skip binary tasks not present in the config
                continue
            col = pd.to_numeric(self.df[task], errors="coerce").dropna()
            n_pos = (col == 1).sum()
            n_neg = (col == 0).sum()
            weights[task] = min(float(n_neg / max(n_pos, 1)), max_weight)
        return weights

    def compute_sample_weights(self, task: str) -> list:
        """Per-sample weights for WeightedRandomSampler. Balances positives/negatives. NaN treated as negative."""
        col = pd.to_numeric(self.df[task], errors="coerce").fillna(0)
        n_pos = (col == 1).sum()
        n_neg = (col == 0).sum()
        w_pos = n_neg / max(n_pos, 1)
        return [float(w_pos) if v == 1 else 1.0 for v in col]

    def compute_union_sample_weights(self, task_alphas: dict) -> list:
        """
        Union sampler that considers several rare tasks at once.
        task_alphas: {task_name: alpha} — weight scale applied to each task's positives
        sample_weight_i = 1 + sum_t( alpha_t * I(label_t==1) / pos_rate_t )
        """
        n = len(self.df)
        weights = np.ones(n, dtype=float)
        for task, alpha in task_alphas.items():
            if task not in self.df.columns:
                continue
            col = pd.to_numeric(self.df[task], errors="coerce").fillna(0).values
            n_pos = (col == 1).sum()
            if n_pos == 0:
                continue
            pos_rate = n_pos / n
            weights += alpha * (col == 1).astype(float) / pos_rate
        return weights.tolist()

    def _compute_feature_stats(self) -> dict:
        stats = {}
        for feat in CONT_FEATURES:
            col = pd.to_numeric(self.df[feat], errors="coerce").values.astype(float)
            valid = col[~np.isnan(col)]
            stats[feat] = {"mean": float(np.mean(valid)), "std": float(np.std(valid) + 1e-6)}
        for feat in BIN_FEATURES:
            col = pd.to_numeric(self.df[feat], errors="coerce").values.astype(float)
            valid = col[~np.isnan(col)]
            stats[feat] = {"mean": float(np.mean(valid)), "std": 1.0}
        if MISMATCH_DIM:                              # mismatch-scalar train statistics (for z-score)
            M = np.array([MISMATCH_FEATS[p] for p in self.df["pat_id"].astype(str)
                          if p in MISMATCH_FEATS], dtype=np.float32)
            mean = np.nanmean(M, 0) if len(M) else np.zeros(MISMATCH_DIM, np.float32)
            std = np.nanstd(M, 0) + 1e-6 if len(M) else np.ones(MISMATCH_DIM, np.float32)
            stats["__mismatch__"] = {"mean": np.nan_to_num(mean), "std": std}
        return stats

    def _get_tabular(self, row, has_dwi: float = 1.0) -> torch.Tensor:
        feats = []
        for feat in CONT_FEATURES:
            val = row[feat]
            val = self.feature_stats[feat]["mean"] if pd.isna(val) else float(val)
            val = (val - self.feature_stats[feat]["mean"]) / self.feature_stats[feat]["std"]
            feats.append(val)
        for feat in BIN_FEATURES:
            val = row[feat]
            val = self.feature_stats[feat]["mean"] if pd.isna(val) else float(val)
            feats.append(val)
        feats.append(has_dwi)
        if MISMATCH_DIM:                              # FLAIR-DWI mismatch (z-score, missing=mean->0)
            st = self.feature_stats["__mismatch__"]
            v = MISMATCH_FEATS.get(str(row["pat_id"]))
            if v is None:
                feats.extend([0.0] * MISMATCH_DIM)
            else:
                z = (np.nan_to_num(v, nan=np.nan) - st["mean"]) / st["std"]
                feats.extend(np.nan_to_num(z).tolist())
        return torch.tensor(feats, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def _get_dwi_path(self, pat_id: str):
        if self.dwi_dir is None:
            return None
        p = os.path.join(self.dwi_dir, f"{pat_id}.nii.gz")
        return p if os.path.exists(p) else None

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        pat_id = str(row["pat_id"])

        dwi_path = self._get_dwi_path(pat_id)
        has_dwi  = 1.0 if dwi_path is not None else 0.0
        sample   = {"image": str(row["t2_path"]), "dwi": dwi_path or str(row["t2_path"])}

        if self.transform:
            sample = self.transform(sample)

        # patients without DWI are filled with zeros
        if dwi_path is None:
            sample["dwi"] = torch.zeros_like(sample["image"])

        labels = {}
        for task in TASK_NAMES:
            raw = row[task] if task in self.df.columns else float("nan")
            if pd.isna(raw):
                val = float("nan")
            elif task in REGRESSION_TASKS:
                val = float(raw) / REG_NORMALIZERS.get(task, 1.0)
            else:
                val = float(raw)
            labels[task] = torch.tensor(val, dtype=torch.float32)

        return {
            "image":   sample["image"],
            "dwi":     sample["dwi"],
            "tabular": self._get_tabular(row, has_dwi=has_dwi),
            **labels,
        }


def multitask_collate_fn(batch):
    """
    Returns:
        images:  (B, 1, D, H, W)  T2
        dwi:     (B, 1, D, H, W)  DWI
        tabular: (B, TAB_FEATURE_DIM)
        labels:  dict of {task_name: (B,)}
    """
    images  = torch.stack([item["image"]   for item in batch])
    dwi     = torch.stack([item["dwi"]     for item in batch])
    tabular = torch.stack([item["tabular"] for item in batch])
    labels  = {task: torch.stack([item[task] for item in batch]) for task in TASK_NAMES}
    return images, dwi, tabular, labels


class TabularOnlyDataset(Dataset):
    """
    Dataset for patients that have only tabular features + outcomes (no image).
    Used only to train tab_only_heads; contributes nothing to the image branch.
    """

    def __init__(self, csv_path: str, clinical_csv_path: str,
                 feature_stats: dict = None):
        main = pd.read_csv(csv_path, dtype={"pat_id": str})
        clin = _load_clinical(clinical_csv_path)
        clin_cols = ["pat_id"] + [c for c in CONT_FEATURES + BIN_FEATURES if c in clin.columns]
        self.df = main.merge(clin[clin_cols], on="pat_id", how="left")
        self.feature_stats = feature_stats or self._compute_feature_stats()

    def _compute_feature_stats(self) -> dict:
        stats = {}
        for feat in CONT_FEATURES:
            col = pd.to_numeric(self.df[feat], errors="coerce").values.astype(float)
            valid = col[~np.isnan(col)]
            stats[feat] = {"mean": float(np.mean(valid)), "std": float(np.std(valid) + 1e-6)}
        for feat in BIN_FEATURES:
            col = pd.to_numeric(self.df[feat], errors="coerce").values.astype(float)
            valid = col[~np.isnan(col)]
            stats[feat] = {"mean": float(np.mean(valid)), "std": 1.0}
        return stats

    def _get_tabular(self, row) -> torch.Tensor:
        feats = []
        for feat in CONT_FEATURES:
            val = row[feat]
            val = self.feature_stats[feat]["mean"] if pd.isna(val) else float(val)
            val = (val - self.feature_stats[feat]["mean"]) / self.feature_stats[feat]["std"]
            feats.append(val)
        for feat in BIN_FEATURES:
            val = row[feat]
            val = self.feature_stats[feat]["mean"] if pd.isna(val) else float(val)
            feats.append(val)
        feats.append(0.0)  # has_dwi: tabular-only patients have no DWI
        if MISMATCH_DIM:                 # tabular-only has no imaging -> mismatch 0
            feats.extend([0.0] * MISMATCH_DIM)
        return torch.tensor(feats, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        labels = {
            task: torch.tensor(
                float("nan") if pd.isna(row[task]) else float(row[task]),
                dtype=torch.float32,
            )
            for task in TASK_NAMES
        }
        return {"tabular": self._get_tabular(row), **labels}


def tabular_only_collate_fn(batch):
    tabular = torch.stack([item["tabular"] for item in batch])
    labels  = {task: torch.stack([item[task] for item in batch]) for task in TASK_NAMES}
    return tabular, labels
