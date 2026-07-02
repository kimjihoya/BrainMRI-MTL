"""Modality attribution — did imaging or clinical drive each prediction?

The fusion head is a single linear layer over the concatenated [image_feature(128) ⊕ clinical_feature(32)]
(dropout is off at eval), so the logit decomposes EXACTLY:

    logit = W_img · image_feature  +  W_clin · clinical_feature  +  bias
          = image_contribution     +  clinical_contribution      +  bias

This is an exact decomposition (not an approximation like SHAP). Per patient we report each modality's
signed contribution to the logit; globally we report mean |contribution| and how often each modality is
the dominant driver. Contributions are averaged over the seed ensemble.

Usage:
    python xai/modality_attribution.py --gpu 0 --n 300 --out_dir results/xai/modality
Run from the repository root.
"""
import os, sys, argparse, numpy as np, pandas as pd
import nibabel as nib
import torch

sys.path.insert(0, os.path.dirname(__file__))
from ft_model import load_operating_model, normalize_clinical  # also puts repo root on sys.path
import paths

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--ckpt_dir", default="checkpoints/mrs_e2e_adamw")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--out_dir", default="results/xai/modality")
ap.add_argument("--dwi_dir", default=paths.DWI_DIR)
ap.add_argument("--clinical_csv", default=paths.CLINICAL_CSV)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out_dir, exist_ok=True)

nets, meta = [], None
for s in range(args.seeds):
    net, meta = load_operating_model(args.ckpt_dir, seed=s, device=device)
    nets.append(net)
CF = meta["tab_features"]
clin = pd.read_csv(args.clinical_csv, dtype={"record_id": str}, encoding="utf-8-sig").set_index("record_id")
oof = np.load("results/mrs/finetune_mrs2_tronly_adamw_oof.npz", allow_pickle=True)
ytrue = {p: int(y) for p, y in zip(oof["pat"].astype(str), oof["y"])}
pids = [p for p in oof["pat"].astype(str)
        if os.path.exists(os.path.join(args.dwi_dir, f"{p}.nii.gz")) and p in clin.index][:args.n]


def znorm(v):
    m = v > 0; z = np.zeros_like(v, np.float32)
    if m.sum() > 100: z[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-6)
    return z


@torch.no_grad()
def contributions(net, x, tab):
    """Split the eval logit into (image, clinical, bias) using the linear head weights."""
    imgf = net.image_feature(x)               # (1,128)
    tabf = net.tab(tab)                        # (1,32)
    lin = net.head[-1]                         # final Linear(160->1)
    W = lin.weight.detach()[0]; b = float(lin.bias.detach()[0])
    img_c = float((W[:128] * imgf[0]).sum())
    clin_c = float((W[128:160] * tabf[0]).sum())
    return img_c, clin_c, b


rows = []
for p in pids:
    vol = znorm(nib.load(os.path.join(args.dwi_dir, f"{p}.nii.gz")).get_fdata().astype(np.float32))
    x = torch.from_numpy(vol[None, None]).float().to(device)
    tab_raw = [pd.to_numeric(clin.loc[p, f], errors="coerce") if f in clin.columns else np.nan for f in CF]
    tab = torch.from_numpy(normalize_clinical(tab_raw, meta)[None]).float().to(device)
    ic, cc, bs = np.mean([contributions(n, x, tab) for n in nets], 0)
    prob = 1 / (1 + np.exp(-(ic + cc + bs)))
    rows.append({"record_id": p, "y_true": ytrue.get(p, -1), "prob": round(prob, 4),
                 "image_contrib": round(ic, 4), "clinical_contrib": round(cc, 4), "bias": round(bs, 4),
                 "driver": "image" if abs(ic) > abs(cc) else "clinical"})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(args.out_dir, "modality_attribution.csv"), index=False)

# ── global summary ──────────────────────────────────────────────────────────
img_imp = df["image_contrib"].abs().mean(); clin_imp = df["clinical_contrib"].abs().mean()
share = img_imp / (img_imp + clin_imp)
print(f"patients={len(df)}")
print(f"mean |image contribution|    = {img_imp:.3f}")
print(f"mean |clinical contribution| = {clin_imp:.3f}")
print(f"-> imaging share of signal   = {share:.1%}  (clinical {1-share:.1%})")
print(f"-> dominant driver: image in {(df['driver']=='image').mean():.1%} of patients")
print(f"saved -> {args.out_dir}/modality_attribution.csv")
