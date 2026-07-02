"""Central path configuration — FILL THESE IN for your environment.

Two path mechanisms exist, by design:
  - Stand-alone scripts (mrs/, xai/, and the non-config preprocessing/extraction
    scripts) import their file-system locations from THIS module.
  - The config-driven pipeline (linear_probe_diagnostic, extract_dwi_embeddings,
    extract_flair_triad, skullstrip_dwi) reads paths from a YAML in configs/ instead.
Keep the placeholders here and in configs/*.yml pointing at the same real directories.

Nothing below points at a real machine; replace `/path/to/...` with your own
directories and checkpoint files.

Scripts are launched from the repository root, so repo-relative paths (e.g. the outcome
CSV and the cached `results/*.npz` features) are left relative on purpose.
"""
import os

# ── Project / data roots ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DATA_ROOT    = "/path/to/data"          # holds DWI_preprocessed/, FLAIR_preprocessed/,
                                        # T2_preprocessed/, all_data.csv

# ── Preprocessed image directories (one .nii.gz per patient, 96^3) ──────────
DWI_DIR           = f"{DATA_ROOT}/DWI_preprocessed"
FLAIR_DIR         = f"{DATA_ROOT}/FLAIR_preprocessed"
T2_DIR            = f"{DATA_ROOT}/T2_preprocessed"
ADC_DIR           = f"{DATA_ROOT}/ADC_data/ADC-data"     # resampled ADC (extraction/preprocessing)
DWI_SKULLSTRIP_DIR = f"{DATA_ROOT}/DWI_skullstrip"       # optional skull-stripped DWI (xai / probes)

# ── Raw archives (only needed to re-run preprocessing from DICOM) ────────────
DWI_ZIP = f"{DATA_ROOT}/DWI_data/DWI_data.zip"

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
