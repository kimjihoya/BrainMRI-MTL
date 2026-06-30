"""SHAP for the clinical branch of the operating mRS model — which admission variables drive it?

A multimodal model makes plain SHAP awkward, so we isolate the clinical contribution: each patient's
image feature is computed once from the DWI encoder, the image feature is held at the dataset mean
(interpretation: "clinical effect at an average scan"), and SHAP attributes the logit to the 32
clinical inputs through the tabular MLP + shared head.

GradientExplainer (a few background samples) gives per-feature SHAP values; we report the global
mean |SHAP| ranking (CSV) and a beeswarm summary plot. SHAP is averaged over the seed ensemble.

Usage:
    python xai/shap_clinical.py --gpu 0 --n 200 --out_dir results/xai/shap
Run from the repository root.
"""
import os, sys, argparse, numpy as np, pandas as pd
import nibabel as nib
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from ft_model import load_operating_model, normalize_clinical

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--ckpt_dir", default="checkpoints/mrs_e2e_adamw")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--n", type=int, default=200, help="patients used for background + explanation")
ap.add_argument("--out_dir", default="results/xai/shap")
ap.add_argument("--dwi_dir", default="/path/to/data/DWI_preprocessed")
ap.add_argument("--clinical_csv", default="/path/to/data/all_data.csv")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out_dir, exist_ok=True)
import shap  # pip install shap

# ── model(s) + clinical ─────────────────────────────────────────────────────
nets, meta = [], None
for s in range(args.seeds):
    net, meta = load_operating_model(args.ckpt_dir, seed=s, device=device)
    nets.append(net)
CF = meta["tab_features"]
clin = pd.read_csv(args.clinical_csv, dtype={"record_id": str}, encoding="utf-8-sig").set_index("record_id")

# patients with a label + a DWI volume (reuse the OOF cohort order)
oof = np.load("results/mrs/finetune_mrs2_radam_oof.npz", allow_pickle=True)
pids = [p for p in oof["pat"].astype(str)
        if os.path.exists(os.path.join(args.dwi_dir, f"{p}.nii.gz")) and p in clin.index][:args.n]
print(f"SHAP on {len(pids)} patients, {len(CF)} clinical features")


def znorm(v):
    m = v > 0; z = np.zeros_like(v, np.float32)
    if m.sum() > 100: z[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-6)
    return z


# normalized clinical matrix
TABraw = np.array([[pd.to_numeric(clin.loc[p, f], errors="coerce") if f in clin.columns else np.nan
                    for f in CF] for p in pids], float)
TAB = normalize_clinical(TABraw, meta)
TABt = torch.from_numpy(TAB).float().to(device)


@torch.no_grad()
def image_features(net):
    feats = []
    for p in pids:
        vol = znorm(nib.load(os.path.join(args.dwi_dir, f"{p}.nii.gz")).get_fdata().astype(np.float32))
        x = torch.from_numpy(vol[None, None]).float().to(device)
        feats.append(net.image_feature(x).cpu().numpy()[0])
    return np.array(feats)


class ClinExplain(nn.Module):
    """logit as a function of (normalized) clinical input, with the image feature fixed to its mean."""
    def __init__(s, net, img_feat_mean):
        super().__init__(); s.net = net
        s.register_buffer("imgf", torch.from_numpy(img_feat_mean).float().view(1, -1))
    def forward(s, clin):
        tabf = s.net.tab(clin)
        return s.net.head(torch.cat([s.imgf.expand(clin.size(0), -1), tabf], 1))


# ── SHAP per seed, then average ─────────────────────────────────────────────
bg = TABt[np.random.RandomState(0).choice(len(pids), min(64, len(pids)), replace=False)]
shap_all = []
for net in nets:
    imgf_mean = image_features(net).mean(0)
    wrapper = ClinExplain(net, imgf_mean).to(device).eval()
    explainer = shap.GradientExplainer(wrapper, bg)
    sv = explainer.shap_values(TABt)
    sv = sv[0] if isinstance(sv, list) else sv
    shap_all.append(np.asarray(sv).reshape(len(pids), len(CF)))
SHAP = np.mean(shap_all, 0)

# ── outputs: ranking CSV + beeswarm ─────────────────────────────────────────
imp = np.abs(SHAP).mean(0)
rank = pd.DataFrame({"feature": CF, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False)
rank.to_csv(os.path.join(args.out_dir, "clinical_shap_importance.csv"), index=False)
np.savez(os.path.join(args.out_dir, "clinical_shap_values.npz"), pat=np.array(pids), feats=np.array(CF),
         shap=SHAP, x=TAB, x_raw=TABraw)
print("\nTop clinical drivers (mean |SHAP|):")
print(rank.head(10).to_string(index=False))

try:
    import matplotlib; matplotlib.use("Agg")
    shap.summary_plot(SHAP, features=TABraw, feature_names=CF, show=False, max_display=15)
    import matplotlib.pyplot as plt
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "clinical_shap_beeswarm.png"), dpi=150)
    print(f"\nsaved -> {args.out_dir}/clinical_shap_importance.csv, clinical_shap_beeswarm.png")
except Exception as e:
    print(f"\n(plot skipped: {e}) saved CSV + npz to {args.out_dir}/")
