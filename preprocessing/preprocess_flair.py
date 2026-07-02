"""
CE-FLAIR DICOM -> nii.gz preprocessing (run on whatever machine holds the raw data)

Exactly the same spec as our T2/DWI preprocessing (preprocess_dwi.py):
  dcm2niix -> resample to 96^3 (order=1) -> save int16 min-max [0,32767].
  WARNING: never z-score here — the training runtime (dataset.py NormalizeIntensityd) does it.

Input layout (NEW_data): one zip per patient
    NEW_data/{pat_id}.zip  ->  contains MRI&MRA/*.dcm (T2 FLAIR series)
    * some patients have 2 FLAIRs (GD + Prohance) -> group by SeriesInstanceUID, pick the series with most slices

Output:
    FLAIR_preprocessed/{pat_id}.nii.gz   (96³, int16)
    -> only this folder needs to be copied to the main machine (~1MB/patient, ~1-2G total)

Dependencies:
    pip install pydicom nibabel numpy scipy
    + the dcm2niix binary (conda install -c conda-forge dcm2niix  or  apt install dcm2niix)

Usage:
    python preprocess_flair.py --in_dir ./NEW_data --out_dir ./FLAIR_preprocessed --workers 4
    # specific patients only:  --pat_ids <pat_id> <pat_id>
"""

import os
import io
import zipfile
import tempfile
import argparse
import subprocess
import traceback
import collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pydicom
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


TARGET_SHAPE = (96, 96, 96)


def resample_to_target(img: nib.Nifti1Image, target=TARGET_SHAPE) -> nib.Nifti1Image:
    """Same as preprocess_dwi.py — order=1 linear resample."""
    data = img.get_fdata(dtype=np.float32)
    factor = [t / s for t, s in zip(target, data.shape)]
    resampled = zoom(data, factor, order=1)
    new_affine = img.affine.copy()
    for i in range(3):
        new_affine[:, i] *= (img.shape[i] / target[i])
    return nib.Nifti1Image(resampled, new_affine)


def pick_series_slices(zf: zipfile.ZipFile):
    """
    Group every .dcm in the zip by SeriesInstanceUID -> pick the series with most slices.
    (collapses the 2-FLAIR [GD/Prohance] case to a single consistent series)
    Returns the chosen series sorted by ImagePositionPatient[2].
    """
    groups = collections.defaultdict(list)
    for name in zf.namelist():
        if not name.lower().endswith(".dcm"):
            continue
        try:
            dcm = pydicom.dcmread(io.BytesIO(zf.read(name)), stop_before_pixels=False)
        except Exception:
            continue
        uid = getattr(dcm, "SeriesInstanceUID", name)
        groups[uid].append(dcm)

    if not groups:
        return None
    # pick the series with most slices (ties broken by 'GD'/'PROHANCE' in SeriesDescription)
    def score(item):
        uid, sl = item
        desc = (getattr(sl[0], "SeriesDescription", "") or "").upper()
        contrast_bonus = 0.5 if ("GD" in desc or "PROHANCE" in desc or "POST" in desc) else 0.0
        return len(sl) + contrast_bonus
    _, slices = max(groups.items(), key=score)

    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        slices.sort(key=lambda x: int(getattr(x, "InstanceNumber", 0)))
    return slices


def process_patient(pat_id: str, in_dir: str, out_dir: str) -> str:
    out_path = os.path.join(out_dir, f"{pat_id}.nii.gz")
    if os.path.exists(out_path):
        return f"SKIP {pat_id}"

    zpath = os.path.join(in_dir, f"{pat_id}.zip")
    if not os.path.exists(zpath):
        return f"FAIL {pat_id}: no zip"

    try:
        zf = zipfile.ZipFile(zpath)
        slices = pick_series_slices(zf)
        if not slices:
            return f"FAIL {pat_id}: no dcm"

        with tempfile.TemporaryDirectory() as tmpdir:
            dcm_dir = os.path.join(tmpdir, "dcm")
            os.makedirs(dcm_dir)
            for i, dcm in enumerate(slices):
                dcm.save_as(os.path.join(dcm_dir, f"slice_{i:04d}.dcm"))

            result = subprocess.run(
                ["dcm2niix", "-z", "y", "-f", "flair", "-o", tmpdir, dcm_dir],
                capture_output=True, text=True,
            )
            nii_files = sorted(Path(tmpdir).glob("flair*.nii.gz"))
            if not nii_files:
                return f"FAIL {pat_id}: no dcm2niix output | {result.stderr[:200]}"

            img = nib.load(str(nii_files[0]))

            # 4D -> 3D (take last volume if multi-volume)
            if img.ndim == 4:
                data = img.get_fdata(dtype=np.float32)[..., -1]
                img = nib.Nifti1Image(data, img.affine)

            if img.shape != TARGET_SHAPE:
                img = resample_to_target(img)

            # save int16 min-max (same as preprocess_dwi.py — z-score happens at runtime)
            data = img.get_fdata(dtype=np.float32)
            mn, mx = data.min(), data.max()
            i16 = ((data - mn) / (mx - mn + 1e-8) * 32767).astype(np.int16)
            nib.save(nib.Nifti1Image(i16, img.affine), out_path)

        return f"OK {pat_id}"

    except Exception:
        return f"FAIL {pat_id}: {traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="./NEW_data", help="folder of per-patient {pat_id}.zip")
    ap.add_argument("--out_dir", default="./FLAIR_preprocessed")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pat_ids", nargs="*", default=None,
                    help="if unset, process every *.zip in in_dir")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pat_ids:
        pat_ids = args.pat_ids
    else:
        pat_ids = sorted(p.stem for p in Path(args.in_dir).glob("*.zip"))

    print(f"to process: {len(pat_ids)} patients | workers={args.workers} | out={args.out_dir}")

    ok = fail = skip = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futs = {exe.submit(process_patient, pid, args.in_dir, args.out_dir): pid
                for pid in pat_ids}
        for i, fut in enumerate(as_completed(futs), 1):
            msg = fut.result()
            if msg.startswith("OK"):
                ok += 1
            elif msg.startswith("SKIP"):
                skip += 1
            else:
                fail += 1
                print(msg)
            if i % 50 == 0 or i == len(pat_ids):
                print(f"  [{i}/{len(pat_ids)}] OK={ok} SKIP={skip} FAIL={fail}")

    print(f"\ndone — OK={ok} SKIP={skip} FAIL={fail}")
    print(f"output: {args.out_dir}  (only this folder needs copying to the main machine)")


if __name__ == "__main__":
    main()
