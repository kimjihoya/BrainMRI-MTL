"""Extract max-pooled 1120-dim FLAIR embeddings through the Triad encoder.
Same pipeline as the DWI extraction (extract_dwi_embeddings.extract); only dwi_dir is swapped to FLAIR.
Output order matches the base cache (tr_pat_id/te_pat_id) -> rows align with DWI-max/labels.
"""
import os, sys, warnings, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "shared")]  # repo root + shared/ modules
from config_utils import load_config
from linear_probe_diagnostic import TARGET_TASKS

CFG = "configs/singletask_end1_binary.yml"
config = load_config(CFG)
FLAIR_DIR = config["data"]["flair_dir"]
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("gpu", {}).get("visible_device", "0"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from model import MultiTaskModel
from dataset import MultiTaskDataset, multitask_collate_fn, get_val_transform

mcfg = config["model"]
model = MultiTaskModel(
    t2_backbone="none", dwi_backbone=mcfg.get("dwi_backbone", "triad"),
    triad_ckpt_path=config.get("triad", {}).get("ckpt_path"),
    task_names=TARGET_TASKS, dwi_feat_dim=mcfg.get("dwi_feat_dim", 256),
    tab_feat_dim=mcfg.get("tab_feat_dim", 128),
).to(device).eval()
enc = model.dwi_enc.encoder
image_size = tuple(config["data"]["size"])

@torch.no_grad()
def extract(csv_path):
    ds = MultiTaskDataset(
        csv_path=csv_path, clinical_csv_path=config["data"]["clinical_csv"],
        transform=get_val_transform(image_size), dwi_dir=FLAIR_DIR,  # <- feed FLAIR
        outcomes_final_csv=config["data"].get("outcomes_final_csv"),
    )
    loader = DataLoader(ds, batch_size=config["data"].get("batch_size", 16),
                        shuffle=False, num_workers=config["data"].get("num_workers", 4),
                        collate_fn=multitask_collate_fn)
    buf = []
    for _, flair, _, _ in loader:
        flair = flair.to(device)
        skips = enc(flair)
        b = skips[0].size(0)
        mx = torch.cat([F.adaptive_max_pool3d(s, 1).view(b, -1) for s in skips], 1)
        buf.append(mx.float().cpu().numpy())
    X = np.concatenate(buf, 0)
    print(f"  {os.path.basename(csv_path)}: n={len(ds)} dim={X.shape[1]}")
    return X

print("[extract] train FLAIR-Triad max ...")
tr = extract(config["data"]["csv_file"])
print("[extract] holdout FLAIR-Triad max ...")
te = extract(config["data"]["val_csv"])
np.savez_compressed("results/flair_triad_max.npz", tr_max=tr, te_max=te)
print(f"saved results/flair_triad_max.npz  tr={tr.shape} te={te.shape}")
