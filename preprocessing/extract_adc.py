"""Extract only the ADC series from each patient zip and save as NIfTI.
Reads DICOM tags inside the raw patient zip (e.g. <pat_id>.zip) to identify the ADC series,
sorts slices, and writes a 3D volume + DICOM geometry (affine) as NIfTI.

Usage:
  python extract_adc.py --zips /path/to/data/sample/*.zip --out results/ADC
  python extract_adc.py --zipdir /path/to/data/original --out results/ADC
"""
import argparse, glob, io, os, re, zipfile
import numpy as np, pydicom, nibabel as nib
from collections import defaultdict


def is_adc(ds):
    desc = str(getattr(ds, "SeriesDescription", "")).lower()
    itype = " ".join(getattr(ds, "ImageType", [])).lower()
    if "eadc" in desc:                       # exclude exponential ADC
        return False
    return ("adc" in desc) or ("adc" in itype)


def find_adc_series(z):
    """Map ADC series in the zip to {uid: [(name, InstanceNumber)]}; pick the one with most slices."""
    series = defaultdict(list)
    meta = {}
    for n in z.namelist():
        if not n.lower().endswith(".dcm"):
            continue
        try:
            ds = pydicom.dcmread(io.BytesIO(z.read(n)), stop_before_pixels=True)
        except Exception:
            continue
        if getattr(ds, "Modality", "") != "MR" or not is_adc(ds):
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "?"))
        series[uid].append(n)
        meta[uid] = str(getattr(ds, "SeriesDescription", ""))
    if not series:
        return None, None
    best = max(series, key=lambda u: len(series[u]))     # most slices = the real ADC
    return series[best], meta[best]


def build_volume(z, names):
    """Sort slices + stack + affine. Prefer ImagePositionPatient, fall back to InstanceNumber."""
    slices = []
    for n in names:
        ds = pydicom.dcmread(io.BytesIO(z.read(n)))
        slices.append(ds)
    # sort: project along the slice-normal direction
    iop = getattr(slices[0], "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    row, col = np.array(iop[:3], float), np.array(iop[3:], float)
    normal = np.cross(row, col)
    def keyf(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            return float(np.dot(np.array(ipp, float), normal))
        return float(getattr(ds, "InstanceNumber", 0))
    slices.sort(key=keyf)
    vol = np.stack([s.pixel_array.astype(np.float32) *
                    float(getattr(s, "RescaleSlope", 1) or 1) +
                    float(getattr(s, "RescaleIntercept", 0) or 0) for s in slices], axis=-1)
    # affine
    ps = [float(x) for x in getattr(slices[0], "PixelSpacing", [1, 1])]
    ipp0 = np.array(getattr(slices[0], "ImagePositionPatient", [0, 0, 0]), float)
    if len(slices) > 1:
        ipp1 = np.array(getattr(slices[-1], "ImagePositionPatient", ipp0 + normal), float)
        sp = np.linalg.norm(ipp1 - ipp0) / max(len(slices) - 1, 1)
    else:
        sp = float(getattr(slices[0], "SliceThickness", 1) or 1)
    aff = np.eye(4)
    aff[:3, 0] = row * ps[0]
    aff[:3, 1] = col * ps[1]
    aff[:3, 2] = normal * (sp or 1)
    aff[:3, 3] = ipp0
    return vol, aff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", nargs="*", default=[])
    ap.add_argument("--zipdir", default=None)
    ap.add_argument("--out", default="results/ADC")
    args = ap.parse_args()
    zips = list(args.zips)
    if args.zipdir:
        zips += sorted(glob.glob(os.path.join(args.zipdir, "*.zip")))
    os.makedirs(args.out, exist_ok=True)
    ok, miss = 0, []
    for zp in zips:
        pid_m = re.search(r"(000000-\d+)", os.path.basename(zp))
        pid = pid_m.group(1) if pid_m else os.path.splitext(os.path.basename(zp))[0]
        try:
            z = zipfile.ZipFile(zp)
            names, desc = find_adc_series(z)
            if not names:
                miss.append(pid); print(f"  [{pid}] no ADC"); continue
            vol, aff = build_volume(z, names)
            outp = os.path.join(args.out, f"{pid}.nii.gz")
            nib.save(nib.Nifti1Image(vol, aff), outp)
            print(f"  [{pid}] '{desc}' {vol.shape} range[{vol.min():.0f},{vol.max():.0f}] → {outp}")
            ok += 1
        except Exception as e:
            miss.append(pid); print(f"  [{pid}] error: {e}")
    print(f"\ndone: {ok} extracted, {len(miss)} failed/missing")
    if miss:
        print("  failed:", miss)


if __name__ == "__main__":
    main()
