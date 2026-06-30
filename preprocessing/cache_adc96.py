"""Resample ADC onto the 96^3 DWI-preprocessing grid and cache it (for CNN input)."""
import glob, os, numpy as np, nibabel as nib, warnings
warnings.filterwarnings("ignore")
from nibabel.processing import resample_from_to

DWI_DIR="/path/to/data/DWI_preprocessed"
ADC_DIR="/path/to/data/ADC_data/ADC-data"
OUT="results/adc96.npz"
dwi_files=sorted(glob.glob(os.path.join(DWI_DIR,"*.nii.gz")))
pats,arrs=[],[]
skip=0
for dp in dwi_files:
    pid=os.path.basename(dp).replace(".nii.gz","")
    ap=os.path.join(ADC_DIR,f"{pid}.nii.gz")
    if not os.path.exists(ap): skip+=1; continue
    try:
        di=nib.load(dp)
        adc=resample_from_to(nib.load(ap),(di.shape,di.affine),order=1).get_fdata().astype(np.float32)
    except Exception:
        skip+=1; continue
    pats.append(pid); arrs.append(adc.astype(np.float16))
np.savez_compressed(OUT,pat_id=np.array(pats),adc=np.array(arrs))
print(f"cached {len(pats)} patients, skipped {skip} -> {OUT}  (shape {np.array(arrs).shape})")
