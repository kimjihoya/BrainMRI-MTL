"""In-lesion ADC scalars — register ADC to the DWI preprocessing space (96^3), then compute ADC
statistics only inside the DWI-hyperintense ROI. Implements 'bright DWI + dark ADC = confirmed acute
core' directly, avoiding whole-brain dilution.
"""
import glob, os, numpy as np, nibabel as nib, warnings
warnings.filterwarnings("ignore")
from nibabel.processing import resample_from_to

DWI_DIR = "/path/to/data/DWI_preprocessed"
ADC_DIR = "/path/to/data/ADC_data/ADC-data"
OUT = "results/adc_lesion_feats.npz"

def zbrain(v):
    m = v > 0
    if m.sum() < 100: return None, None
    z = np.zeros_like(v); z[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-6)
    return z, m

def feats(pid):
    dp = os.path.join(DWI_DIR, f"{pid}.nii.gz")
    ap = os.path.join(ADC_DIR, f"{pid}.nii.gz")
    if not (os.path.exists(dp) and os.path.exists(ap)): return None
    dwi_img = nib.load(dp); adc_img = nib.load(ap)
    dwi = dwi_img.get_fdata().astype(np.float32)
    # resample ADC onto the DWI grid (anatomical registration)
    try:
        adc = resample_from_to(adc_img, (dwi_img.shape, dwi_img.affine), order=1).get_fdata().astype(np.float32)
    except Exception:
        return None
    dz, dm = zbrain(dwi)
    if dz is None: return None
    brain = dm & (adc > 50) & (adc < 3000)
    if brain.sum() < 500: return None
    f = {}
    # ADC inside the DWI-hyperintense lesion ROI (several thresholds)
    for thr in [1.5, 2.0, 2.5]:
        roi = brain & (dz > thr)
        nvox = int(roi.sum())
        f[f"roi{thr}_nvox"] = float(nvox)
        if nvox >= 10:
            ra = adc[roi]
            f[f"roi{thr}_adc_mean"] = float(ra.mean())
            f[f"roi{thr}_adc_min"]  = float(np.percentile(ra, 5))
            f[f"roi{thr}_adc_lo620_frac"] = float((ra < 620).mean())
            # if the lesion is truly acute, ROI ADC < normal-brain ADC -> ratio
            normal = adc[brain & (dz < 0.5)]
            f[f"roi{thr}_adc_ratio"] = float(ra.mean() / (normal.mean() + 1e-6)) if normal.size else 1.0
        else:
            f[f"roi{thr}_adc_mean"]=np.nan; f[f"roi{thr}_adc_min"]=np.nan
            f[f"roi{thr}_adc_lo620_frac"]=np.nan; f[f"roi{thr}_adc_ratio"]=np.nan
    # confirmed acute-core volume = DWI-hyperintense ∩ ADC-low (the key mismatch intersection)
    for dth, ath in [(2.0, 620), (2.0, 720), (1.5, 720)]:
        core = brain & (dz > dth) & (adc < ath)
        f[f"core_d{dth}_a{ath}_vol"] = float(core.sum())
        f[f"core_d{dth}_a{ath}_frac"] = float(core.sum() / brain.sum())
    return f

def main():
    files = sorted(glob.glob(os.path.join(ADC_DIR, "*.nii.gz")))
    pats, rows, names = [], [], None
    skip = 0
    for fp in files:
        pid = os.path.basename(fp).replace(".nii.gz", "")
        d = feats(pid)
        if d is None: skip += 1; continue
        if names is None: names = list(d.keys())
        pats.append(pid); rows.append([d.get(k, np.nan) for k in names])
    X = np.array(rows, float)
    os.makedirs("results", exist_ok=True)
    np.savez(OUT, pat_id=np.array(pats), feats=X, names=np.array(names))
    print(f"extracted {len(pats)} patients, {len(names) if names else 0} scalars, skipped {skip} -> {OUT}")
    print("scalars:", names)

if __name__ == "__main__":
    main()
