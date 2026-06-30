"""Shared loader for the operating mRS model (used by the XAI scripts).

Rebuilds the end-to-end fine-tuned FTNet (Triad DWI encoder + clinical fusion) and loads a saved
seed checkpoint from `checkpoints/mrs_e2e_<optim>/`. Kept separate from the training script so the
explainability tools can import the model without running the training pipeline.

The FTNet definition below MUST match codes/mrs/train_mrs_endtoend.py.
"""
import os, sys, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from model import _load_triad

TRIAD_CKPT = "/path/to/Triad/weight/Triad-PlainConvUNet-MAE.pth"   # fill in (see paths.py)


class FTNet(nn.Module):
    """Triad DWI encoder (last stage trainable) ⊕ clinical fusion. Mirrors the training script."""
    def __init__(s, enc, n_tab, unfreeze_last=1, drop=0.5, use_t2=False, t2dim=768):
        super().__init__()
        s.stages = enc.stages; s.nstage = len(s.stages); s.ufz = unfreeze_last; s.use_t2 = use_t2
        for i, st in enumerate(s.stages):
            for p in st.parameters(): p.requires_grad = i >= s.nstage - unfreeze_last
        s.img_dim = 256 + 320 + 320
        s.img_bn = nn.BatchNorm1d(s.img_dim)
        s.img_fc = nn.Sequential(nn.Dropout(drop), nn.Linear(s.img_dim, 128), nn.ReLU())
        s.tab = nn.Sequential(nn.Linear(n_tab, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU())
        h = 128 + 32
        if use_t2:
            s.t2 = nn.Sequential(nn.BatchNorm1d(t2dim), nn.Dropout(drop), nn.Linear(t2dim, 64), nn.ReLU())
            h += 64
        s.head = nn.Sequential(nn.Dropout(drop), nn.Linear(h, 1))

    def _skips(s, x, grad_all=False):
        """Run the encoder, returning the last 3 skips. grad_all=True keeps grad on every stage
        (needed for Grad-CAM, since the operating forward wraps frozen stages in no_grad)."""
        sk = []
        for i, st in enumerate(s.stages):
            if (i < s.nstage - s.ufz) and not grad_all:
                with torch.no_grad(): x = st(x)
            else:
                x = st(x)
            sk.append(x)
        return sk[-3:]

    def image_feature(s, x, grad_all=False, return_maps=False):
        maps = s._skips(x, grad_all=grad_all)
        b = maps[0].size(0)
        pooled = torch.cat([F.adaptive_avg_pool3d(t, 1).view(b, -1) for t in maps], 1)
        feat = s.img_fc(s.img_bn(pooled))
        return (feat, maps) if return_maps else feat

    def forward(s, x, tab, t2=None, grad_all=False):
        feats = [s.image_feature(x, grad_all=grad_all), s.tab(tab)]
        if s.use_t2 and t2 is not None: feats.append(s.t2(t2))
        return s.head(torch.cat(feats, 1))


def load_operating_model(ckpt_dir, seed=0, device="cuda", n_tab=32, unfreeze=1, use_t2=False):
    """Load one seed checkpoint of the operating mRS model + its metadata (tab normalization)."""
    enc = _load_triad(TRIAD_CKPT).encoder
    net = FTNet(enc, n_tab, unfreeze_last=unfreeze, use_t2=use_t2).to(device)
    state = torch.load(os.path.join(ckpt_dir, f"model_seed{seed}.pt"), map_location=device)
    net.load_state_dict(state)
    net.eval()
    meta = json.load(open(os.path.join(ckpt_dir, "metadata.json")))
    return net, meta


def normalize_clinical(tab_raw, meta):
    """Apply the checkpoint's saved train-set mean/std to raw clinical features."""
    mu = np.array(meta["tab_mean"], float); sd = np.array(meta["tab_std"], float)
    mu = np.where(np.isnan(mu), 0, mu); sd = np.where((sd == 0) | np.isnan(sd), 1e-6, sd)
    return np.nan_to_num((np.asarray(tab_raw, float) - mu) / sd)
