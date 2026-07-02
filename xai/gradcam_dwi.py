"""3D Grad-CAM for the operating mRS model — where on the DWI did the model look?

Computes Grad-CAM on the Triad DWI encoder of the end-to-end fine-tuned model: the gradient of the
predicted poor-outcome logit w.r.t. an encoder feature map is global-average-pooled into channel
weights, combined with the activations, ReLU'd, and upsampled to the 96^3 DWI grid. The heatmap is
saved as NIfTI (same affine as the DWI) so it can be overlaid on the lesion in any viewer.

Target layer: the shallowest of the 3 skips the model pools (stage nstage-3, 12^3) — the best trade
of spatial detail vs. semantic content. The encoder is re-run with gradients on every stage
(grad_all=True), since the operating forward wraps frozen stages in no_grad.

Usage:
    python xai/gradcam_dwi.py --gpu 0 --pat <pat_id> --out_dir results/xai/gradcam
    python xai/gradcam_dwi.py --gpu 0 --pat auto --topk 12   # most confident poor-outcome cases
Run from the repository root.
"""
import os, sys, argparse, numpy as np, pandas as pd
import nibabel as nib
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from ft_model import load_operating_model, normalize_clinical  # also puts repo root on sys.path
import paths

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--ckpt_dir", default="checkpoints/mrs_e2e_adamw")
ap.add_argument("--seeds", type=int, default=3, help="ensemble over this many seed checkpoints")
ap.add_argument("--pat", default="auto", help="patient id, or 'auto' for top-confident cases")
ap.add_argument("--topk", type=int, default=8, help="how many patients when --pat auto")
ap.add_argument("--out_dir", default="results/xai/gradcam")
ap.add_argument("--dwi_dir", default=paths.DWI_DIR)
ap.add_argument("--clinical_csv", default=paths.CLINICAL_CSV)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out_dir, exist_ok=True)

# ── load the operating model (seed ensemble) + clinical features ────────────
nets, meta = [], None
for s in range(args.seeds):
    net, meta = load_operating_model(args.ckpt_dir, seed=s, device=device)
    nets.append(net)
CF = meta["tab_features"]
clin = pd.read_csv(args.clinical_csv, dtype={"record_id": str}, encoding="utf-8-sig").set_index("record_id")


def znorm(v):
    m = v > 0; z = np.zeros_like(v, np.float32)
    if m.sum() > 100: z[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-6)
    return z


def load_inputs(pid):
    img = nib.load(os.path.join(args.dwi_dir, f"{pid}.nii.gz"))
    vol = znorm(img.get_fdata().astype(np.float32))
    x = torch.from_numpy(vol[None, None]).float().to(device)
    tab_raw = [pd.to_numeric(clin.loc[pid, f], errors="coerce") if (pid in clin.index and f in clin.columns)
               else np.nan for f in CF]
    tab = torch.from_numpy(normalize_clinical(tab_raw, meta)[None]).float().to(device)
    return x, tab, img


def gradcam_one(net, x, tab):
    """Grad-CAM map (96^3, [0,1]) + the poor-outcome probability, for one model."""
    net.zero_grad()
    x = x.detach().requires_grad_(True)   # input must require grad so activations carry grad
    # capture the shallowest pooled skip with grad retained
    _, maps = net.image_feature(x, grad_all=True, return_maps=True)
    act = maps[0]            # stage nstage-3, e.g. 12^3 x 256
    act.retain_grad()
    feat = net.img_fc(net.img_bn(torch.cat(
        [F.adaptive_avg_pool3d(t, 1).view(1, -1) for t in maps], 1)))
    logit = net.head(torch.cat([feat, net.tab(tab)], 1))
    prob = torch.sigmoid(logit).item()
    logit.backward()
    grad = act.grad                                   # (1,C,d,d,d)
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)  # GAP over space -> channel weights
    cam = F.relu((weights * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(96, 96, 96), mode="trilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, prob


def run(pid):
    x, tab, img = load_inputs(pid)
    cams, probs = [], []
    for net in nets:
        cam, prob = gradcam_one(net, x, tab)
        cams.append(cam); probs.append(prob)
    cam = np.mean(cams, 0); prob = float(np.mean(probs))
    aff = img.affine
    nib.save(nib.Nifti1Image(cam.astype(np.float32), aff), os.path.join(args.out_dir, f"{pid}_gradcam.nii.gz"))
    nib.save(nib.Nifti1Image(img.get_fdata().astype(np.float32), aff), os.path.join(args.out_dir, f"{pid}_dwi.nii.gz"))
    print(f"  {pid}: P(poor)={prob:.3f}  -> {pid}_gradcam.nii.gz (overlay on {pid}_dwi.nii.gz)")
    return prob


# ── pick patients ───────────────────────────────────────────────────────────
if args.pat != "auto":
    pids = [args.pat]
else:
    # rank available patients by predicted probability, take the most confident poor-outcome cases
    oof = np.load("results/mrs/finetune_mrs2_tronly_adamw_oof.npz", allow_pickle=True)
    order = np.argsort(oof["oof"])[::-1]
    avail = {os.path.basename(p).replace(".nii.gz", "") for p in os.listdir(args.dwi_dir)} \
        if os.path.isdir(args.dwi_dir) else set(oof["pat"].astype(str))
    pids = [p for p in oof["pat"][order].astype(str) if p in avail][:args.topk]

print(f"Grad-CAM on {len(pids)} patient(s) -> {args.out_dir}")
for pid in pids:
    try: run(pid)
    except Exception as e: print(f"  {pid}: skipped ({e})")
print("done. View each *_gradcam.nii.gz as a heatmap overlay on *_dwi.nii.gz.")
