"""
DWI DICOM -> nii.gz preprocessing

- extract the per-patient b=1000 trace volume from DWI_data.zip (nested)
- convert to nii.gz with dcm2niix
- resample to 96x96x96, save int16 (saves disk)
- write /path/to/data/DWI_preprocessed/{pat_id}.nii.gz

Usage:
    python preprocess_dwi.py                        # only patients in the split CSV (default)
    python preprocess_dwi.py --workers 4
    python preprocess_dwi.py --csv my.csv           # a specific CSV
"""

import os
import io
import zipfile
import tempfile
import argparse
import subprocess
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pydicom
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


TARGET_SHAPE = (96, 96, 96)
DWI_ZIP = "/path/to/data/DWI_data/DWI_data.zip"


def is_b1000(dcm: pydicom.Dataset) -> bool:
    seq = getattr(dcm, "SequenceName", "") or ""
    return "b1000" in seq.lower()


def load_sorted_slices(inner_zip: zipfile.ZipFile, b1000_only: bool = True):
    slices = []
    for name in inner_zip.namelist():
        if not name.endswith(".dcm"):
            continue
        dcm = pydicom.dcmread(io.BytesIO(inner_zip.read(name)), stop_before_pixels=False)
        if b1000_only and not is_b1000(dcm):
            continue
        slices.append(dcm)

    if not slices:
        return None

    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except Exception:
        slices.sort(key=lambda x: int(getattr(x, "InstanceNumber", 0)))

    return slices


def resample_to_target(img: nib.Nifti1Image, target=TARGET_SHAPE) -> nib.Nifti1Image:
    data = img.get_fdata(dtype=np.float32)
    factor = [t / s for t, s in zip(target, data.shape)]
    resampled = zoom(data, factor, order=1)
    new_affine = img.affine.copy()
    for i in range(3):
        new_affine[:, i] *= (img.shape[i] / target[i])
    return nib.Nifti1Image(resampled, new_affine)


def process_patient(pat_id: str, out_dir: str) -> str:
    out_path = os.path.join(out_dir, f"{pat_id}.nii.gz")
    if os.path.exists(out_path):
        return f"SKIP {pat_id}"

    try:
        outer = zipfile.ZipFile(DWI_ZIP)
        inner_name = f"DWI_data/{pat_id}.zip"
        if inner_name not in outer.namelist():
            return f"FAIL {pat_id}: not in zip"

        inner_bytes = io.BytesIO(outer.read(inner_name))
        inner = zipfile.ZipFile(inner_bytes)

        # prefer b=1000 slices, otherwise take all
        slices = load_sorted_slices(inner, b1000_only=True)
        if not slices:
            slices = load_sorted_slices(inner, b1000_only=False)
        if not slices:
            return f"FAIL {pat_id}: no slices"

        with tempfile.TemporaryDirectory() as tmpdir:
            dcm_dir = os.path.join(tmpdir, "dcm")
            os.makedirs(dcm_dir)
            for i, dcm in enumerate(slices):
                dcm.save_as(os.path.join(dcm_dir, f"slice_{i:04d}.dcm"))

            result = subprocess.run(
                ["dcm2niix", "-z", "y", "-f", "dwi", "-o", tmpdir, dcm_dir],
                capture_output=True, text=True,
            )

            nii_files = sorted(Path(tmpdir).glob("dwi*.nii.gz"))
            if not nii_files:
                return f"FAIL {pat_id}: dcm2niix no output | {result.stderr[:200]}"

            # if multiple files, prefer b1000/trace
            chosen = nii_files[0]
            if len(nii_files) > 1:
                for f in nii_files:
                    if "b1000" in f.name.lower() or "trace" in f.name.lower():
                        chosen = f
                        break

            img = nib.load(str(chosen))

            # 4D -> 3D (last volume)
            if img.ndim == 4:
                data = img.get_fdata(dtype=np.float32)[..., -1]
                img = nib.Nifti1Image(data, img.affine)

            if img.shape != TARGET_SHAPE:
                img = resample_to_target(img)

            # save normalized as int16
            data = img.get_fdata(dtype=np.float32)
            mn, mx = data.min(), data.max()
            i16 = ((data - mn) / (mx - mn + 1e-8) * 32767).astype(np.int16)
            nib.save(nib.Nifti1Image(i16, img.affine), out_path)

        return f"OK {pat_id}"

    except Exception:
        return f"FAIL {pat_id}: {traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="/path/to/data/DWI_preprocessed")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pat_ids", nargs="*", default=None)
    parser.add_argument("--csv",     default=None, help="CSV with a pat_id column")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pat_ids:
        pat_ids = args.pat_ids
    elif args.csv:
        import pandas as pd
        pat_ids = pd.read_csv(args.csv, dtype={"pat_id": str})["pat_id"].tolist()
    else:
        import pandas as pd
        ids = set()
        for p in [
            "./data/csvs/split_train.csv",
            "./data/csvs/split_test.csv",
        ]:
            if os.path.exists(p):
                ids.update(pd.read_csv(p, dtype={"pat_id": str})["pat_id"].tolist())
        pat_ids = sorted(ids)

    print(f"to process: {len(pat_ids)} patients  |  workers={args.workers}  |  out={args.out_dir}")

    ok = fail = skip = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(process_patient, pid, args.out_dir): pid for pid in pat_ids}
        for i, future in enumerate(as_completed(futures), 1):
            msg = future.result()
            if msg.startswith("OK"):
                ok += 1
            elif msg.startswith("SKIP"):
                skip += 1
            else:
                fail += 1
                print(msg)
            if i % 50 == 0 or i == len(pat_ids):
                print(f"  [{i}/{len(pat_ids)}]  OK={ok}  SKIP={skip}  FAIL={fail}")

    print(f"\ndone — OK={ok}  SKIP={skip}  FAIL={fail}")
    print(f"output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
