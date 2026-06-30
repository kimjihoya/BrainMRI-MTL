"""Extract true acute-core scalars from ADC (an END1 lever).
Low ADC = cytotoxic edema = acute infarct core. From the whole-brain ADC distribution (no threshold)
derive low-ADC volume/fraction/extremes as scalars (no DWI registration needed, ADC alone).
"""
import glob, os, numpy as np, nibabel as nib

ADC_DIR = "/path/to/data/ADC_data/ADC-data"
OUT = "results/adc_core_feats.npz"

# acute-core thresholds (x10^-6 mm^2/s). Literature: <~620 = core, ~620-820 = at risk.
THRESH = [520, 620, 720, 820]

def brain_mask(v):
    # ADC>0 & plausible brain range (CSF<3000, noise>50). Clip the tails.
    return (v > 50) & (v < 3000)

def feats_for(v):
    m = brain_mask(v)
    if m.sum() < 1000:
        return None
    br = v[m]
    n = br.size
    f = {}
    # low-ADC volume (voxel count) & fraction — proxy for acute-core size
    for t in THRESH:
        f[f"adc_lo{t}_vol"]  = float((br < t).sum())
        f[f"adc_lo{t}_frac"] = float((br < t).mean())
    # extremes / distribution
    f["adc_min"]   = float(np.percentile(br, 1))    # minimum (1% to avoid noise)
    f["adc_p5"]    = float(np.percentile(br, 5))
    f["adc_p10"]   = float(np.percentile(br, 10))
    f["adc_median"]= float(np.median(br))
    f["adc_mean"]  = float(br.mean())
    f["adc_std"]   = float(br.std())
    # low-ADC concentration: mean of the darkest 1% voxels (catches small cores too)
    lo = np.percentile(br, 1)
    f["adc_darkest1pct_mean"] = float(br[br <= lo].mean()) if (br <= lo).any() else float(lo)
    return f

def main():
    files = sorted(glob.glob(os.path.join(ADC_DIR, "*.nii.gz")))
    pats, rows, names = [], [], None
    skip = 0
    for fp in files:
        pid = os.path.basename(fp).replace(".nii.gz", "")
        try:
            v = nib.load(fp).get_fdata().astype(np.float32)
        except Exception:
            skip += 1; continue
        d = feats_for(v)
        if d is None:
            skip += 1; continue
        if names is None:
            names = list(d.keys())
        pats.append(pid); rows.append([d[k] for k in names])
    X = np.array(rows, float)
    os.makedirs("results", exist_ok=True)
    np.savez(OUT, pat_id=np.array(pats), feats=X, names=np.array(names))
    print(f"extracted {len(pats)} patients, {len(names)} scalars, skipped {skip} -> {OUT}")
    print("scalars:", names)

if __name__ == "__main__":
    main()
