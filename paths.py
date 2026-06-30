"""Central path configuration — FILL THESE IN for your environment.

Every script in this repo reads its file-system locations from here (or has the same
placeholders inline). Nothing below points at a real machine; replace `/path/to/...`
with your own directories and checkpoint files.

Scripts are launched from the repository root, so repo-relative paths (e.g. the outcome
CSV and the cached `results/*.npz` features) are left relative on purpose.
"""
import os

# ── Project / data roots ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DATA_ROOT    = "/path/to/data"          # holds DWI_preprocessed/, FLAIR_preprocessed/,
                                        # T2_preprocessed/, all_data.csv

# ── Preprocessed image directories (one .nii.gz per patient, 96^3) ──────────
DWI_DIR   = f"{DATA_ROOT}/DWI_preprocessed"
FLAIR_DIR = f"{DATA_ROOT}/FLAIR_preprocessed"
T2_DIR    = f"{DATA_ROOT}/T2_preprocessed"

# ── Tabular data ────────────────────────────────────────────────────────────
CLINICAL_CSV = f"{DATA_ROOT}/all_data.csv"               # admission clinical features
OUTCOMES_CSV = "data/csvs/outcomes_final.csv"            # single source of truth for labels (repo-relative)

# ── Pretrained backbone weights (see README → Acknowledgements) ─────────────
TRIAD_CKPT    = "/path/to/Triad/weight/Triad-PlainConvUNet-MAE.pth"   # DWI encoder
BRAINIAC_CKPT = "/path/to/BrainIAC/checkpoints/BrainIAC.ckpt"         # T2 encoder
BRAINMVP_CKPT = "/path/to/BrainMVP/checkpoint.pth"                    # optional alternate

# ── Cached features (produced by codes/extraction/, repo-relative) ──────────
LINEAR_PROBE_FEATS = "results/linear_probe_features.npz"   # foundation embeddings + clinical tab
POOLING_DWI_FEATS  = "results/pooling_dwi_features.npz"    # DWI pooled embeddings (max/gap/std/gapmax)
FLAIR_MISMATCH     = "results/flair_mismatch_feats.npz"    # engineered FLAIR-DWI mismatch scalars
