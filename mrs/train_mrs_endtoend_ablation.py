"""Hyperparameter ablation for the end-to-end multimodal mRS fine-tune,
generalized across the 3 targets (dis_mrs / mrs1y / mrs3mo).

This is a copy of train_mrs_endtoend.py (the pristine operating-model script,
left untouched so its reported numbers stay reproducible) parameterized with
--target and with each run's result appended to a shared CSV instead of just
an npz, so a sweep across configs can be tracked in one table.

tr-only CV (holdout-test excluded) is the ONLY mode this script runs -- this
mirrors the project's honest model-selection protocol (never select configs by
peeking at holdout). Same architecture/training code as the operating script;
only the label column and reported output differ.

Usage: python codes/mrs/train_mrs_endtoend_ablation.py --target mrs1y --optim radam --gpu 1
"""
import os, sys, time, glob, json, csv, warnings, argparse
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True, choices=["dis_mrs", "mrs1y", "mrs3mo"])
ap.add_argument("--gpu", default="0")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--unfreeze", type=int, default=1)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--enc_lr", type=float, default=1e-5)
ap.add_argument("--head_lr", type=float, default=1e-3)
ap.add_argument("--drop", type=float, default=0.6)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--swa", type=int, default=1)
ap.add_argument("--swa_k", type=int, default=10)
ap.add_argument("--tta", type=int, default=1)
ap.add_argument("--cosine", type=int, default=1)
ap.add_argument("--optim", default="adamw", choices=["adamw", "sgd", "adam", "radam", "nadam", "rmsprop"])
ap.add_argument("--tag", default="")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "shared")]
from model import _load_triad
import paths

DWI_DIR = paths.DWI_DIR
CLINICAL_CSV = paths.CLINICAL_CSV
TRIAD_CKPT = paths.TRIAD_CKPT
LP_FEATS = "results/linear_probe_features.npz"  # only used for the fixed te_pat_id holdout set
OUT_CSV = "results/ablation/e2e_hparam_results.csv"
OUT_NPZ_DIR = "results/ablation/npz"
os.makedirs(OUT_NPZ_DIR, exist_ok=True)

TARGET_COL = {"dis_mrs": "dis_mrs", "mrs1y": "mrs1y", "mrs3mo": "mrs3mo"}
dev = "cuda" if torch.cuda.is_available() else "cpu"


def build_opt(enc_p, head_p):
    groups = [{"params": enc_p, "lr": args.enc_lr}, {"params": head_p, "lr": args.head_lr}]
    o = args.optim
    if o == "adamw": return torch.optim.AdamW(groups, weight_decay=args.wd)
    if o == "adam": return torch.optim.Adam(groups, weight_decay=args.wd)
    if o == "radam": return torch.optim.RAdam(groups, weight_decay=args.wd)
    if o == "nadam": return torch.optim.NAdam(groups, weight_decay=args.wd)
    if o == "rmsprop": return torch.optim.RMSprop(groups, weight_decay=args.wd)
    if o == "sgd": return torch.optim.SGD(groups, momentum=0.9, nesterov=True, weight_decay=args.wd)
    raise ValueError(o)


clin = pd.read_csv(CLINICAL_CSV, dtype={"record_id": str}, encoding="utf-8-sig").set_index("record_id")
v = pd.to_numeric(clin[TARGET_COL[args.target]], errors="coerce"); v = v.where(v != 9)
yl = pd.Series(np.nan, index=clin.index); yl[v >= 3] = 1.0; yl[v <= 2] = 0.0
CF = ["age", "male", "sbp", "dbp", "ini_nih", "pre_mrs", "toast", "hx_htn", "hx_dm", "hx_af", "hx_hl",
      "hx_str", "hx_chd", "smok", "ekg_af", "ecg", "wbc", "hb", "hct", "plt", "pt", "bun", "cr", "crp",
      "fbs", "i_glu", "ha1c", "tc", "tg", "hdl", "ldl", "n_ssctri"]

pats, ys, vpaths = [], [], []
for dp in sorted(glob.glob(os.path.join(DWI_DIR, "*.nii.gz"))):
    pid = os.path.basename(dp).replace(".nii.gz", "")
    if pid not in yl.index or np.isnan(yl[pid]) or pid not in clin.index: continue
    pats.append(pid); ys.append(float(yl[pid])); vpaths.append(dp)
ys = np.array(ys); pats = np.array(pats)
TAB = np.array([[pd.to_numeric(clin.loc[p, f], errors="coerce") if f in clin.columns else np.nan
                  for f in CF] for p in pats], float)
cfg = f"target={args.target} optim={args.optim} swa={args.swa} drop={args.drop} wd={args.wd} unfreeze={args.unfreeze}"
print(f"cohort={len(ys)} pos={int(ys.sum())} ({ys.mean():.1%})  {cfg}", flush=True)


def znorm(v):
    m = v > 0; z = np.zeros_like(v, np.float32)
    if m.sum() > 100: z[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-6)
    return z


print("loading DWI volumes...", flush=True)
t0 = time.time()
VOL = np.zeros((len(ys), 1, 96, 96, 96), np.float16)
for i, p in enumerate(vpaths): VOL[i, 0] = znorm(nib.load(p).get_fdata().astype(np.float32))
print(f"volumes loaded ({time.time()-t0:.0f}s)", flush=True)


def augment(xb):
    if np.random.rand() < 0.5: xb = torch.flip(xb, [2])
    if np.random.rand() < 0.5: xb = torch.flip(xb, [3])
    if np.random.rand() < 0.5: xb = torch.flip(xb, [4])
    nr = np.random.randint(0, 4)
    if nr: xb = torch.rot90(xb, nr, [2, 3])
    if np.random.rand() < 0.3: xb = xb + torch.randn_like(xb) * 0.1
    if np.random.rand() < 0.3: xb = xb * float(np.random.uniform(0.8, 1.2))
    return xb


class FTNet(nn.Module):
    def __init__(s, enc, n_tab, unfreeze_last=1, drop=0.5):
        super().__init__()
        s.stages = enc.stages; s.nstage = len(s.stages); s.ufz = unfreeze_last
        for i, st in enumerate(s.stages):
            for p in st.parameters(): p.requires_grad = i >= s.nstage - unfreeze_last
        s.img_dim = 256 + 320 + 320
        s.img_bn = nn.BatchNorm1d(s.img_dim)
        s.img_fc = nn.Sequential(nn.Dropout(drop), nn.Linear(s.img_dim, 128), nn.ReLU())
        s.tab = nn.Sequential(nn.Linear(n_tab, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU())
        s.head = nn.Sequential(nn.Dropout(drop), nn.Linear(160, 1))

    def encode(s, x):
        sk = []
        for i, st in enumerate(s.stages):
            if i < s.nstage - s.ufz:
                with torch.no_grad(): x = st(x)
            else: x = st(x)
            sk.append(x)
        b = sk[0].size(0)
        return torch.cat([F.adaptive_avg_pool3d(t, 1).view(b, -1) for t in sk[-3:]], 1)

    def forward(s, x, tab):
        feats = [s.img_fc(s.img_bn(s.encode(x))), s.tab(tab)]
        return s.head(torch.cat(feats, 1))


def run_fold(tr, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    enc = _load_triad(TRIAD_CKPT).encoder
    net = FTNet(enc, TAB.shape[1], args.unfreeze, args.drop).to(dev)
    enc_p = [p for p in net.stages.parameters() if p.requires_grad]
    head_p = [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith("stages")]
    opt = build_opt(enc_p, head_p)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs) if args.cosine else None
    scaler = torch.cuda.amp.GradScaler()
    pos_w = torch.tensor([(ys[tr] == 0).sum() / max((ys[tr] == 1).sum(), 1)], device=dev, dtype=torch.float32)
    mu = np.nanmean(TAB[tr], 0); mu = np.where(np.isnan(mu), 0, mu); sd = np.nanstd(TAB[tr], 0) + 1e-6
    tabn = np.nan_to_num((TAB - mu) / sd)
    swa_states = []

    def set_train():
        net.train()
        for i, stg in enumerate(net.stages):
            if i < net.nstage - net.ufz:
                for m in stg.modules():
                    if isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)): m.eval()

    for ep in range(args.epochs):
        set_train(); idx = np.random.permutation(tr)
        for s0 in range(0, len(idx), 8):
            bi = idx[s0:s0 + 8]
            if len(bi) < 2: continue
            xb = augment(torch.from_numpy(VOL[bi].astype(np.float32)).to(dev))
            yb = torch.tensor(ys[bi], dtype=torch.float32, device=dev)[:, None]
            tb = torch.tensor(tabn[bi], dtype=torch.float32, device=dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                out = net(xb, tb); loss = F.binary_cross_entropy_with_logits(out, yb, pos_weight=pos_w)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if sched: sched.step()
        if args.swa and ep >= args.epochs - args.swa_k:
            swa_states.append({k: v.detach().clone().float() for k, v in net.state_dict().items()})
    if args.swa and swa_states:
        avg = {k: sum(s[k] for s in swa_states) / len(swa_states) for k in swa_states[0]}
        net.load_state_dict(avg)
    flips = [[], [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]] if args.tta else [[]]
    net.eval(); ps = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for s0 in range(0, len(te), 8):
            bi = te[s0:s0 + 8]
            x0 = torch.from_numpy(VOL[bi].astype(np.float32)).to(dev)
            tb = torch.tensor(tabn[bi], dtype=torch.float32, device=dev)
            acc = 0
            for fl in flips:
                xb = torch.flip(x0, fl) if fl else x0
                acc = acc + torch.sigmoid(net(xb, tb)).float()
            ps.append((acc / len(flips)).cpu().numpy().ravel())
    return np.concatenate(ps)


rng = np.random.default_rng(0)
def ci(y, p, n=3000):
    a = []; idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), True)
        if y[b].sum() < 2 or (1 - y[b]).sum() < 2: continue
        a.append(roc_auc_score(y[b], p[b]))
    return np.percentile(a, [2.5, 97.5])


# tr-only CV: holdout-test excluded (correct model-selection protocol)
lp = np.load(LP_FEATS, allow_pickle=True)
te_ids = set(lp["te_pat_id"].astype(str))
pool = np.where(np.array([p not in te_ids for p in pats]))[0]
print(f"[tr-only OOF] holdout-test excluded: pool={len(pool)} pos={int(ys[pool].sum())}", flush=True)
ypool = ys[pool]
oof = np.zeros(len(pool)); cnt = np.zeros(len(pool))
t_run0 = time.time()
for seed in range(args.seeds):
    for fi, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.arange(len(pool)), ypool)):
        oof[te] += run_fold(pool[tr], pool[te], seed); cnt[te] += 1
    print(f"  [seed{seed}] OOF={roc_auc_score(ypool, oof/np.maximum(cnt,1)):.3f}  "
          f"({time.time()-t_run0:.0f}s elapsed)", flush=True)
oof /= np.maximum(cnt, 1)
a = roc_auc_score(ypool, oof); lo, hi = ci(ypool, oof); apv = average_precision_score(ypool, oof)
runtime = time.time() - t_run0
print(f"\n★ [{args.target}] OOF AUROC={a:.3f} [{lo:.3f},{hi:.3f}] AP={apv:.3f}  ({cfg})  "
      f"runtime={runtime:.0f}s", flush=True)

npz_path = f"{OUT_NPZ_DIR}/{args.target}{args.tag}_oof.npz"
np.savez(npz_path, pat=pats[pool], y=ypool, oof=oof, auroc=a, ci=[lo, hi])

row = {"target": args.target, "optim": args.optim, "swa": args.swa, "unfreeze": args.unfreeze,
       "drop": args.drop, "wd": args.wd, "epochs": args.epochs, "seeds": args.seeds,
       "n": len(ypool), "n_pos": int(ypool.sum()), "auroc": round(a, 4), "ci_lo": round(lo, 4),
       "ci_hi": round(hi, 4), "ap": round(apv, 4), "runtime_s": round(runtime, 1), "npz": npz_path,
       "tag": args.tag}
new_file = not os.path.exists(OUT_CSV)
with open(OUT_CSV, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if new_file: w.writeheader()
    w.writerow(row)
print(f"appended -> {OUT_CSV}", flush=True)
