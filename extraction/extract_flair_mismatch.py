"""Extract low-dimensional FLAIR-DWI mismatch scalar features.
- z-score normalize each DWI/FLAIR volume over nonzero voxels (same as the NormalizeIntensityd pipeline).
- registration-free features (volume / intensity distribution) are primary; ROI-overlap features are auxiliary.
Output: results/flair_mismatch_feats.npz (pat_id, feats, names)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, nibabel as nib, os, glob
from tqdm import tqdm

DWI_DIR = "/path/to/data/DWI_preprocessed"
FLAIR_DIR = "/path/to/data/FLAIR_preprocessed"

lp = np.load("results/linear_probe_features.npz", allow_pickle=True)
pat = np.concatenate([lp["tr_pat_id"], lp["te_pat_id"]]).astype(str)

def zload(path):
    d = nib.load(path).get_fdata().astype(np.float32)
    m = d > 0
    if m.sum() < 100:
        return None, None
    z = np.zeros_like(d)
    z[m] = (d[m] - d[m].mean()) / (d[m].std() + 1e-6)
    return z, m

THRS = [1.5, 2.0, 2.5]
names, rows, kept = None, [], []
for pid in tqdm(pat):
    fd = os.path.join(DWI_DIR, f"{pid}.nii.gz")
    ff = os.path.join(FLAIR_DIR, f"{pid}.nii.gz")
    if not (os.path.exists(fd) and os.path.exists(ff)):
        continue
    Dz, Dm = zload(fd); Fz, Fm = zload(ff)
    if Dz is None or Fz is None:
        continue
    brain = Dm.sum()
    feat, nm = [], []
    # --- registration-free: volume / intensity distribution ---
    for t in THRS:
        dvol = (Dz > t).sum()
        fvol = (Fz > t).sum()
        feat += [dvol / brain, fvol / brain]
        nm += [f"dwi_hi_frac_{t}", f"flair_hi_frac_{t}"]
        # mismatch volume ratio: FLAIR lesion vs DWI lesion (smaller = FLAIR-negative = hyperacute)
        feat += [(dvol - fvol) / (dvol + 1.0), fvol / (dvol + 1.0)]
        nm += [f"vol_mismatch_{t}", f"flair_dwi_volratio_{t}"]
    # DWI/FLAIR intensity tail (lesion severity)
    for p in [90, 95, 99]:
        feat += [np.percentile(Dz[Dm], p), np.percentile(Fz[Fm], p)]
        nm += [f"dwi_p{p}", f"flair_p{p}"]
    # --- ROI overlap (registration-dependent, auxiliary): FLAIR signal inside the DWI lesion ---
    roi = Dz > 2.0
    if roi.sum() >= 20:
        feat += [Fz[roi].mean(), np.percentile(Fz[roi], 90),
                 Fz[roi].mean() / (Dz[roi].mean() + 1e-6)]
        # Dice of DWI-lesion ∩ FLAIR-lesion (mismatch = low)
        froi = Fz > 2.0
        inter = (roi & froi).sum()
        feat += [2 * inter / (roi.sum() + froi.sum() + 1e-6)]
    else:
        feat += [0.0, 0.0, 0.0, 0.0]
    nm += ["roi_flair_mean", "roi_flair_p90", "roi_flair_dwi_ratio", "roi_lesion_dice"]
    if names is None:
        names = nm
    rows.append(feat); kept.append(pid)

X = np.array(rows, dtype=np.float32)
kept = np.array(kept)
np.savez("results/flair_mismatch_feats.npz", pat_id=kept, feats=X, names=np.array(names))
print(f"saved {X.shape} for {len(kept)} patients, {len(names)} features")
print("features:", names)
