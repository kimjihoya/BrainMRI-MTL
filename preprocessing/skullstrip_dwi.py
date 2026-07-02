"""
DWI skull-strip experiment: remove the skull (Otsu + largest-connected-component mask), then
extract frozen Triad features -> does the END1 probe improve over whole-head (0.558)?
Aligned to our cache order (pat_id). Needs a GPU. Usage: python skullstrip_dwi.py --gpu 1
"""
import argparse, os, sys, numpy as np, torch, torch.nn.functional as F, nibabel as nib
from scipy import ndimage
from skimage.filters import threshold_otsu

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "shared")]  # repo root + shared/ modules

DWI_DIR = "/path/to/data/DWI_preprocessed"   # overridden from config in main()


def brain_mask(vol):
    """DWI b1000: brain parenchyma is bright, skull is dark -> Otsu + largest component + fill + closing."""
    v = vol.astype(np.float32)
    pos = v[v > 0]
    if pos.size < 100:
        return v > 0
    try:
        th = threshold_otsu(pos)
    except Exception:
        th = np.percentile(pos, 50)
    m = v > th
    m = ndimage.binary_closing(m, iterations=2)
    lbl, n = ndimage.label(m)
    if n >= 1:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        m = lbl == (np.argmax(sizes) + 1)
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_dilation(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    return m


def load_ss(pat_id):
    arr = nib.load(f"{DWI_DIR}/{pat_id}.nii.gz").get_fdata().astype(np.float32)
    m = brain_mask(arr)
    masked = arr * m
    br = masked[m]
    mu, sd = br.mean(), br.std() + 1e-6
    z = np.where(m, (masked - mu) / sd, 0.0).astype(np.float32)
    return z, m.mean()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", default="1")
    args = ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from config_utils import load_config
    from model import MultiTaskModel
    cfg = load_config("configs/best/router_slim_v2_frozen.yml")
    global DWI_DIR
    DWI_DIR = cfg["data"].get("dwi_dir", DWI_DIR)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskModel(t2_backbone="none", dwi_backbone="triad",
                           triad_ckpt_path=cfg["triad"]["ckpt_path"],
                           task_names=["end1_binary"]).to(dev).eval()
    enc = model.dwi_enc.encoder

    lp = np.load("results/linear_probe_features.npz", allow_pickle=True)
    order = {"tr": lp["tr_pat_id"].astype(str), "te": lp["te_pat_id"].astype(str)}
    out = {}; nzfrac = []
    for split, pats in order.items():
        feats = []
        for pid in pats:
            z, nz = load_ss(pid); nzfrac.append(nz)
            t = torch.from_numpy(z)[None, None].to(dev)
            skips = enc(t)
            f = torch.cat([F.adaptive_max_pool3d(s, 1).view(1, -1) for s in skips], 1)
            feats.append(f.float().cpu().numpy())
        out[split] = np.concatenate(feats, 0)
        print(f"  {split}: {out[split].shape}")
    np.savez_compressed("results/dwi_skullstrip_features.npz",
                        tr_ss=out["tr"], te_ss=out["te"])
    print(f"mean brain-mask fraction: {np.mean(nzfrac)*100:.1f}% (whole-head is ~91%)")
    print("saved -> results/dwi_skullstrip_features.npz")


if __name__ == "__main__":
    main()
