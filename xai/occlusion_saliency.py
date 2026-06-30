"""Occlusion saliency for the operating mRS model — faithful to the training distribution.

Grad-CAM on this model is misleading: it uses global average pooling (decision = mean activation,
not localized) and the training input is whole-head DWI (not skull-stripped), so Grad-CAM gets
dominated by high-intensity skull/scalp edges that do NOT actually drive the prediction.

Occlusion saliency sidesteps both problems: on the ACTUAL whole-head input the model was trained on,
slide an occlusion cube, zero it, and measure the drop in P(poor). Brain regions that matter light up;
the skull doesn't (occluding it barely changes P). This is the correct saliency for a GAP-global model.

Usage:
    python xai_occlusion.py --gpu 0 --pat auto --topk 6 --win 16 --stride 8
Run from the repository root. Input defaults to the training distribution (whole-head DWI_preprocessed).
"""
import os, sys, glob, argparse, numpy as np, pandas as pd, nibabel as nib, torch
sys.path.insert(0, os.path.dirname(__file__))
from ft_model import load_operating_model, normalize_clinical

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--ckpt_dir", default="checkpoints/mrs_e2e_adamw")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--pat", default="auto")
ap.add_argument("--topk", type=int, default=6)
ap.add_argument("--win", type=int, default=16, help="occlusion cube size (voxels)")
ap.add_argument("--stride", type=int, default=8)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--out_dir", default="results/xai/occlusion")
ap.add_argument("--dwi_dir", default="/path/to/data/DWI_preprocessed")  # training distribution (compute input)
ap.add_argument("--brain_dir", default="", help="skull-stripped dir: restrict occlusion to brain + use as display bg")
ap.add_argument("--clinical_csv", default="/path/to/data/all_data.csv")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out_dir, exist_ok=True)

nets, meta = [], None
for s in range(args.seeds):
    net, meta = load_operating_model(args.ckpt_dir, seed=s, device=device); nets.append(net)
CF = meta["tab_features"]
clin = pd.read_csv(args.clinical_csv, dtype={"record_id": str}, encoding="utf-8-sig").set_index("record_id")


def znorm(v):
    m = v > 0; z = np.zeros_like(v, np.float32)
    if m.sum() > 100: z[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-6)
    return z


@torch.no_grad()
def predict_batch(vol_batch, tab):
    """vol_batch (B,1,96,96,96) -> mean P(poor) over the seed ensemble, per item."""
    xb = torch.from_numpy(vol_batch).float().to(device)
    tb = tab.expand(xb.size(0), -1)
    p = torch.zeros(xb.size(0), device=device)
    for net in nets: p += torch.sigmoid(net(xb, tb)).squeeze(1)
    return (p / len(nets)).cpu().numpy()


def run(pid):
    img = nib.load(os.path.join(args.dwi_dir, f"{pid}.nii.gz"))
    base = znorm(img.get_fdata().astype(np.float32))           # compute input = whole-head (deployment)
    # occlusion is restricted to brain (skull occlusion is provably P-invariant); display bg = brain
    if args.brain_dir:
        bvol = nib.load(os.path.join(args.brain_dir, f"{pid}.nii.gz")).get_fdata().astype(np.float32)
        region = bvol > 0; disp = bvol
    else:
        region = base != 0; disp = img.get_fdata().astype(np.float32)
    tab_raw = [pd.to_numeric(clin.loc[pid, f], errors="coerce") if f in clin.columns else np.nan for f in CF]
    tab = torch.from_numpy(normalize_clinical(tab_raw, meta)[None]).float().to(device)
    p_full = float(predict_batch(base[None, None], tab)[0])

    W, S = args.win, args.stride
    centers = list(range(0, 96, S))
    sal = np.zeros((96, 96, 96), np.float32); cnt = np.zeros((96, 96, 96), np.float32)
    # build occlusion boxes that overlap the brain region
    boxes = []
    for x in centers:
        for y in centers:
            for z in centers:
                xs, ys, zs = slice(x, min(x + W, 96)), slice(y, min(y + W, 96)), slice(z, min(z + W, 96))
                if region[xs, ys, zs].any(): boxes.append((xs, ys, zs))
    # batched forward
    for i in range(0, len(boxes), args.batch):
        chunk = boxes[i:i + args.batch]
        vb = np.repeat(base[None, None], len(chunk), 0)
        for k, (xs, ys, zs) in enumerate(chunk): vb[k, 0, xs, ys, zs] = 0.0
        dp = p_full - predict_batch(vb, tab)            # drop in P when this region is removed
        for (xs, ys, zs), d in zip(chunk, dp):
            sal[xs, ys, zs] += d; cnt[xs, ys, zs] += 1
    sal = np.where(cnt > 0, sal / np.maximum(cnt, 1), 0.0)
    sal = np.clip(sal, 0, None)                         # keep only positive evidence (removing it lowers P)
    sal = sal / (sal.max() + 1e-8)
    nib.save(nib.Nifti1Image(sal.astype(np.float32), img.affine), os.path.join(args.out_dir, f"{pid}_occ.nii.gz"))
    nib.save(nib.Nifti1Image(disp.astype(np.float32), img.affine), os.path.join(args.out_dir, f"{pid}_dwi.nii.gz"))
    print(f"  {pid}: P(poor)={p_full:.3f}, {len(boxes)} occlusions -> {pid}_occ.nii.gz")


if args.pat != "auto":
    pids = [args.pat]
else:
    oof = np.load("results/mrs/finetune_mrs2_radam_oof.npz", allow_pickle=True)
    order = np.argsort(oof["oof"])[::-1]
    avail = {os.path.basename(p).replace(".nii.gz", "") for p in os.listdir(args.dwi_dir)}
    pids = [p for p in oof["pat"][order].astype(str) if p in avail][:args.topk]

print(f"Occlusion saliency on {len(pids)} patient(s), win={args.win} stride={args.stride} -> {args.out_dir}")
for pid in pids:
    try: run(pid)
    except Exception as e: print(f"  {pid}: skipped ({e})")
print("done.")
