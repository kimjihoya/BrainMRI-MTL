"""mRS 3-month poor outcome (mrs3mo >= 3) — OPERATING MODEL (end-to-end multimodal).

Triad DWI encoder (last stage fine-tuned) + admission clinical, intermediate (feature-level) fusion.

  DWI [96^3] -> Triad encoder (last stage trainable) -> gap-pool last 3 skips -> MLP -> 128-d
  clinical(32) -> MLP -> 32-d   [optional T2(BrainIAC,768) -> 64-d]   concat -> head -> logit

CORRECT PROTOCOL — pick the model on TR-ONLY cross-validation, never on the holdout:
  --tr_only 1 (DEFAULT)  : 5x5 OOF on holdout-train (542) only; the test set is fully excluded.
                           This is the model-SELECTION metric (optimizer, hyper-params).
  --holdout 1            : after the config is locked, train on tr (542), predict the unseen
                           test (144) exactly ONCE -> the reported independent estimate.
  Selecting by holdout = selection bias (a small 33-positive test overfits to noise).

Operating config (chosen by tr-only CV): --optim adamw --swa 1 --wd 5e-2 --drop 0.6 --unfreeze 1 --seeds 3
  -> tr-only CV 0.810 / independent holdout 0.765.
  (An earlier RAdam 0.843 was picked by peeking at the holdout and is discarded.)

Levers (all flags): --optim {adamw,radam,sgd,...}, --swa, --drop/--wd, TTA 8-flip + N-seed ensemble,
  --use_t2 (frozen T2 branch; hurt generalization -> off). --save_final dumps the operating checkpoint.
  Run from the repository root."""
import os, glob, numpy as np, pandas as pd, warnings, argparse, sys
warnings.filterwarnings("ignore")
ap=argparse.ArgumentParser()
ap.add_argument("--gpu",default="0"); ap.add_argument("--epochs",type=int,default=40)
ap.add_argument("--unfreeze",type=int,default=1); ap.add_argument("--seeds",type=int,default=3)
ap.add_argument("--enc_lr",type=float,default=1e-5); ap.add_argument("--head_lr",type=float,default=1e-3)
ap.add_argument("--drop",type=float,default=0.5); ap.add_argument("--wd",type=float,default=1e-2)
ap.add_argument("--use_t2",type=int,default=0); ap.add_argument("--swa",type=int,default=0); ap.add_argument("--swa_k",type=int,default=10)
ap.add_argument("--tta",type=int,default=1); ap.add_argument("--cosine",type=int,default=1)
ap.add_argument("--holdout",type=int,default=0); ap.add_argument("--tag",default="")
ap.add_argument("--save_final",type=int,default=0)
ap.add_argument("--optim",default="adamw",choices=["adamw","sgd","adam","radam","nadam","rmsprop"])
ap.add_argument("--tr_only",type=int,default=1,help="OOF on holdout-train only (test excluded) = clean internal CV. DEFAULT. set 0 only for legacy full-cohort OOF")
ap.add_argument("--rep_oof",type=float,default=0.0); ap.add_argument("--rep_holdout",type=float,default=0.0)
args=ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
import torch, torch.nn as nn, torch.nn.functional as F
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0,".")
from model import _load_triad
TRIAD_CKPT="/path/to/Triad/weight/Triad-PlainConvUNet-MAE.pth"
DWI_DIR="/path/to/data/DWI_preprocessed"
dev="cuda" if torch.cuda.is_available() else "cpu"

def build_opt(enc_p,head_p):
    groups=[{"params":enc_p,"lr":args.enc_lr},{"params":head_p,"lr":args.head_lr}]
    o=args.optim
    if o=="adamw":   return torch.optim.AdamW(groups,weight_decay=args.wd)
    if o=="adam":    return torch.optim.Adam(groups,weight_decay=args.wd)
    if o=="radam":   return torch.optim.RAdam(groups,weight_decay=args.wd)
    if o=="nadam":   return torch.optim.NAdam(groups,weight_decay=args.wd)
    if o=="rmsprop": return torch.optim.RMSprop(groups,weight_decay=args.wd)
    if o=="sgd":     return torch.optim.SGD(groups,momentum=0.9,nesterov=True,weight_decay=args.wd)
    raise ValueError(o)

clin=pd.read_csv("/path/to/data/all_data.csv",dtype={"record_id":str},encoding="utf-8-sig").set_index("record_id")
v=pd.to_numeric(clin["mrs3mo"],errors="coerce"); v=v.where(v!=9)
yl=pd.Series(np.nan,index=clin.index); yl[v>=3]=1.0; yl[v<=2]=0.0
CF=["age","male","sbp","dbp","ini_nih","pre_mrs","toast","hx_htn","hx_dm","hx_af","hx_hl",
    "hx_str","hx_chd","smok","ekg_af","ecg","wbc","hb","hct","plt","pt","bun","cr","crp",
    "fbs","i_glu","ha1c","tc","tg","hdl","ldl","n_ssctri"]
# T2 frozen embedding (BrainIAC), keyed by patient id
lp=np.load("results/linear_probe_features.npz",allow_pickle=True)
T2emb=np.concatenate([lp["tr_t2"],lp["te_t2"]],0)
t2pat=np.concatenate([lp["tr_pat_id"],lp["te_pat_id"]]).astype(str)
T2MAP={p:T2emb[i] for i,p in enumerate(t2pat)}; T2DIM=T2emb.shape[1]

pats,ys,paths=[],[],[]
for dp in sorted(glob.glob(os.path.join(DWI_DIR,"*.nii.gz"))):
    pid=os.path.basename(dp).replace(".nii.gz","")
    if pid not in yl.index or np.isnan(yl[pid]) or pid not in clin.index: continue
    pats.append(pid); ys.append(float(yl[pid])); paths.append(dp)
ys=np.array(ys); pats=np.array(pats)
TAB=np.array([[pd.to_numeric(clin.loc[p,f],errors="coerce") if f in clin.columns else np.nan for f in CF] for p in pats],float)
T2=np.array([T2MAP[p] if p in T2MAP else np.full(T2DIM,np.nan) for p in pats],float)
cfg=f"use_t2={args.use_t2} swa={args.swa} drop={args.drop} wd={args.wd} unfreeze={args.unfreeze}"
print(f"cohort={len(ys)} pos={int(ys.sum())} ({ys.mean():.1%})  {cfg}",flush=True)

def znorm(v):
    m=v>0; z=np.zeros_like(v,np.float32)
    if m.sum()>100: z[m]=(v[m]-v[m].mean())/(v[m].std()+1e-6)
    return z
print("loading DWI volumes...",flush=True)
VOL=np.zeros((len(ys),1,96,96,96),np.float16)
for i,p in enumerate(paths): VOL[i,0]=znorm(nib.load(p).get_fdata().astype(np.float32))
print("volumes loaded",flush=True)

def augment(xb):
    if np.random.rand()<0.5: xb=torch.flip(xb,[2])
    if np.random.rand()<0.5: xb=torch.flip(xb,[3])
    if np.random.rand()<0.5: xb=torch.flip(xb,[4])
    nr=np.random.randint(0,4)
    if nr: xb=torch.rot90(xb,nr,[2,3])
    if np.random.rand()<0.3: xb=xb+torch.randn_like(xb)*0.1
    if np.random.rand()<0.3: xb=xb*float(np.random.uniform(0.8,1.2))
    return xb

class FTNet(nn.Module):
    def __init__(s,enc,n_tab,unfreeze_last=1,drop=0.5,use_t2=False,t2dim=768):
        super().__init__()
        s.stages=enc.stages; s.nstage=len(s.stages); s.ufz=unfreeze_last; s.use_t2=use_t2
        for i,st in enumerate(s.stages):
            for p in st.parameters(): p.requires_grad = i>=s.nstage-unfreeze_last
        s.img_dim=256+320+320
        s.img_bn=nn.BatchNorm1d(s.img_dim)
        s.img_fc=nn.Sequential(nn.Dropout(drop),nn.Linear(s.img_dim,128),nn.ReLU())
        s.tab=nn.Sequential(nn.Linear(n_tab,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,32),nn.ReLU())
        h=128+32
        if use_t2:
            s.t2=nn.Sequential(nn.BatchNorm1d(t2dim),nn.Dropout(drop),nn.Linear(t2dim,64),nn.ReLU())
            h+=64
        s.head=nn.Sequential(nn.Dropout(drop),nn.Linear(h,1))
    def encode(s,x):
        sk=[]
        for i,st in enumerate(s.stages):
            if i<s.nstage-s.ufz:
                with torch.no_grad(): x=st(x)
            else: x=st(x)
            sk.append(x)
        b=sk[0].size(0)
        return torch.cat([F.adaptive_avg_pool3d(t,1).view(b,-1) for t in sk[-3:]],1)
    def forward(s,x,tab,t2=None):
        feats=[s.img_fc(s.img_bn(s.encode(x))),s.tab(tab)]
        if s.use_t2 and t2 is not None: feats.append(s.t2(t2))
        return s.head(torch.cat(feats,1))

def run_fold(tr,te,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    enc=_load_triad(TRIAD_CKPT).encoder
    net=FTNet(enc,TAB.shape[1],args.unfreeze,args.drop,bool(args.use_t2),T2DIM).to(dev)
    enc_p=[p for p in net.stages.parameters() if p.requires_grad]
    head_p=[p for n,p in net.named_parameters() if p.requires_grad and not n.startswith("stages")]
    opt=build_opt(enc_p,head_p)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs) if args.cosine else None
    scaler=torch.cuda.amp.GradScaler()
    pos_w=torch.tensor([(ys[tr]==0).sum()/max((ys[tr]==1).sum(),1)],device=dev,dtype=torch.float32)
    mu=np.nanmean(TAB[tr],0); mu=np.where(np.isnan(mu),0,mu); sd=np.nanstd(TAB[tr],0)+1e-6
    tabn=np.nan_to_num((TAB-mu)/sd)
    mt=np.nanmean(T2[tr],0); mt=np.where(np.isnan(mt),0,mt); st=np.nanstd(T2[tr],0)+1e-6
    t2n=np.nan_to_num((np.where(np.isnan(T2),mt,T2)-mt)/st)
    swa_states=[]
    def set_train():
        net.train()
        for i,stg in enumerate(net.stages):
            if i<net.nstage-net.ufz:
                for m in stg.modules():
                    if isinstance(m,(nn.BatchNorm3d,nn.InstanceNorm3d)): m.eval()
    for ep in range(args.epochs):
        set_train(); idx=np.random.permutation(tr)
        for s0 in range(0,len(idx),8):
            bi=idx[s0:s0+8]
            if len(bi)<2: continue
            xb=augment(torch.from_numpy(VOL[bi].astype(np.float32)).to(dev))
            yb=torch.tensor(ys[bi],dtype=torch.float32,device=dev)[:,None]
            tb=torch.tensor(tabn[bi],dtype=torch.float32,device=dev)
            t2b=torch.tensor(t2n[bi],dtype=torch.float32,device=dev) if args.use_t2 else None
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                out=net(xb,tb,t2b); loss=F.binary_cross_entropy_with_logits(out,yb,pos_weight=pos_w)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if sched: sched.step()
        if args.swa and ep>=args.epochs-args.swa_k:
            swa_states.append({k:v.detach().clone().float() for k,v in net.state_dict().items()})
    if args.swa and swa_states:  # manual SWA: average weights over the last K epochs
        avg={k:sum(s[k] for s in swa_states)/len(swa_states) for k in swa_states[0]}
        net.load_state_dict(avg)
    flips=[[],[2],[3],[4],[2,3],[2,4],[3,4],[2,3,4]] if args.tta else [[]]
    net.eval(); ps=[]
    with torch.no_grad(), torch.cuda.amp.autocast():
        for s0 in range(0,len(te),8):
            bi=te[s0:s0+8]
            x0=torch.from_numpy(VOL[bi].astype(np.float32)).to(dev)
            tb=torch.tensor(tabn[bi],dtype=torch.float32,device=dev)
            t2b=torch.tensor(t2n[bi],dtype=torch.float32,device=dev) if args.use_t2 else None
            acc=0
            for fl in flips:
                xb=torch.flip(x0,fl) if fl else x0
                acc=acc+torch.sigmoid(net(xb,tb,t2b)).float()
            ps.append((acc/len(flips)).cpu().numpy().ravel())
    return np.concatenate(ps)

rng=np.random.default_rng(0)
def ci(y,p,n=3000):
    a=[];idx=np.arange(len(y))
    for _ in range(n):
        b=rng.choice(idx,len(idx),True)
        if y[b].sum()<2 or (1-y[b]).sum()<2: continue
        a.append(roc_auc_score(y[b],p[b]))
    return np.percentile(a,[2.5,97.5])

if args.holdout:
    te_ids=set(lp["te_pat_id"].astype(str)); is_te=np.array([p in te_ids for p in pats])
    tr_idx=np.where(~is_te)[0]; te_idx=np.where(is_te)[0]
    print(f"\n=== holdout: train on {len(tr_idx)} ({int(ys[tr_idx].sum())} pos) -> predict unseen {len(te_idx)} ({int(ys[te_idx].sum())} pos) ===",flush=True)
    pe=np.zeros(len(te_idx))
    for seed in range(args.seeds):
        pe+=run_fold(tr_idx,te_idx,seed); print(f"  seed{seed} done",flush=True)
    pe/=args.seeds; yte=ys[te_idx]; aH=roc_auc_score(yte,pe); lo,hi=ci(yte,pe)
    print(f"\n★ holdout AUROC={aH:.3f} [{lo:.3f},{hi:.3f}] AP={average_precision_score(yte,pe):.3f}  ({cfg})")
    np.savez(f"results/mrs/finetune_mrs2{args.tag}_holdout.npz",pat=pats[te_idx],y=yte,pred=pe,auroc=aH)
    sys.exit(0)

if args.save_final:
    import json
    out=f"checkpoints/mrs_e2e_{args.optim}"
    os.makedirs(out,exist_ok=True)
    allidx=np.arange(len(ys))
    mu=np.nanmean(TAB,0); mu=np.where(np.isnan(mu),0,mu); sd=np.nanstd(TAB,0)+1e-6
    mt=np.nanmean(T2,0); mt=np.where(np.isnan(mt),0,mt); st=np.nanstd(T2,0)+1e-6
    # train on the full cohort + SWA, save one state_dict per seed (deployed model)
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        enc=_load_triad(TRIAD_CKPT).encoder
        net=FTNet(enc,TAB.shape[1],args.unfreeze,args.drop,bool(args.use_t2),T2DIM).to(dev)
        enc_p=[p for p in net.stages.parameters() if p.requires_grad]
        head_p=[p for n,p in net.named_parameters() if p.requires_grad and not n.startswith("stages")]
        opt=build_opt(enc_p,head_p)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
        scaler=torch.cuda.amp.GradScaler()
        pos_w=torch.tensor([(ys==0).sum()/max((ys==1).sum(),1)],device=dev,dtype=torch.float32)
        tabn=np.nan_to_num((TAB-mu)/sd)
        swa_states=[]
        for ep in range(args.epochs):
            net.train()
            for i,stg in enumerate(net.stages):
                if i<net.nstage-net.ufz:
                    for m in stg.modules():
                        if isinstance(m,(nn.BatchNorm3d,nn.InstanceNorm3d)): m.eval()
            ii=np.random.permutation(allidx)
            for s0 in range(0,len(ii),8):
                bi=ii[s0:s0+8]
                if len(bi)<2: continue
                xb=augment(torch.from_numpy(VOL[bi].astype(np.float32)).to(dev))
                yb=torch.tensor(ys[bi],dtype=torch.float32,device=dev)[:,None]
                tb=torch.tensor(tabn[bi],dtype=torch.float32,device=dev)
                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    loss=F.binary_cross_entropy_with_logits(net(xb,tb,None),yb,pos_weight=pos_w)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step()
            if ep>=args.epochs-args.swa_k:
                swa_states.append({k:v.detach().clone().float() for k,v in net.state_dict().items()})
        avg={k:sum(s[k] for s in swa_states)/len(swa_states) for k in swa_states[0]}
        net.load_state_dict(avg)
        torch.save(net.state_dict(),f"{out}/model_seed{seed}.pt")
        print(f"  saved seed{seed}",flush=True)
    meta={"target":"mrs3mo>=3 (3-month poor outcome)","model":f"Triad DWI end-to-end fine-tune (last stage) + clinical fusion, {args.optim.upper()}+SWA+strong-reg, 3seed+TTA",
          "optimizer":args.optim,"use_t2":bool(args.use_t2),"swa":bool(args.swa),"swa_k":args.swa_k,"drop":args.drop,"wd":args.wd,"unfreeze":args.unfreeze,
          "enc_lr":args.enc_lr,"head_lr":args.head_lr,"epochs":args.epochs,"tta":bool(args.tta),
          "cohort_n":int(len(ys)),"n_pos":int(ys.sum()),"tab_features":CF,"tab_mean":mu.tolist(),"tab_std":sd.tolist(),
          "oof_auroc":args.rep_oof,"holdout_auroc":args.rep_holdout,
          "note":"AdamW selected by tr-only CV (0.810); independent holdout 0.765. Earlier RAdam 0.843 was selection bias (picked via holdout) and discarded."}
    json.dump(meta,open(f"{out}/metadata.json","w"),ensure_ascii=False,indent=2)
    print(f"saved operating checkpoint -> {out}/ (per-seed weights + metadata)")
    sys.exit(0)

# OOF cohort: full 686, or tr-only (holdout-test excluded -> clean internal validation)
if args.tr_only:
    te_ids=set(lp["te_pat_id"].astype(str)); pool=np.where(np.array([p not in te_ids for p in pats]))[0]
    print(f"[tr-only OOF] holdout-test fully excluded: pool={len(pool)} pos={int(ys[pool].sum())}",flush=True)
else:
    pool=np.arange(len(ys))
ypool=ys[pool]
oof=np.zeros(len(pool)); cnt=np.zeros(len(pool))
for seed in range(args.seeds):
    for fi,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.arange(len(pool)),ypool)):
        oof[te]+=run_fold(pool[tr],pool[te],seed); cnt[te]+=1
    print(f"  [seed{seed}] OOF={roc_auc_score(ypool,oof/np.maximum(cnt,1)):.3f}",flush=True)
oof/=np.maximum(cnt,1)
a=roc_auc_score(ypool,oof); lo,hi=ci(ypool,oof); apv=average_precision_score(ypool,oof)
ys=ypool; pats=pats[pool]
print(f"\n★ OOF AUROC={a:.3f} [{lo:.3f},{hi:.3f}] AP={apv:.3f}  ({cfg}){' [TR-ONLY]' if args.tr_only else ''}")
np.savez(f"results/mrs/finetune_mrs2{args.tag}_oof.npz",pat=pats,y=ys,oof=oof,auroc=a,ci=[lo,hi])
print(f"saved -> results/mrs/finetune_mrs2{args.tag}_oof.npz")
