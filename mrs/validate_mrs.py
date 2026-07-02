"""Strict, fully-separated validation: the test split is excluded from OOF and used once.
internal : 5x5 OOF on holdout-train only + permutation test (is the signal real within train?).
final    : train on all of holdout-train -> predict the never-seen holdout-test once (no leakage).
targets  : mrs (frozen deep fusion, DWI gap-pool PCA30 + clinical) / dementia (DWI max-pool alone)."""
import os, sys, numpy as np, pandas as pd, warnings, argparse
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> paths.py
import paths
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

ap=argparse.ArgumentParser(); ap.add_argument("--target",default="mrs"); ap.add_argument("--nperm",type=int,default=200)
args=ap.parse_args()
rng=np.random.default_rng(0)
lp=np.load("results/linear_probe_features.npz",allow_pickle=True)
pp=np.load("results/pooling_dwi_features.npz",allow_pickle=True)
TAB=np.concatenate([lp["tr_tab"],lp["te_tab"]],0)
ntr=lp["tr_tab"].shape[0]
pat=np.concatenate([lp["tr_pat_id"],lp["te_pat_id"]]).astype(str)
oc=pd.read_csv("data/csvs/outcomes_final.csv",dtype={"record_id":str},encoding="utf-8-sig")
oc.columns=[c.strip().replace("﻿","") for c in oc.columns]; oc=oc.set_index("record_id")
clin=pd.read_csv(paths.CLINICAL_CSV,dtype={"record_id":str},encoding="utf-8-sig").set_index("record_id")

if args.target=="mrs":
    DWI=np.concatenate([pp["tr_gap"],pp["te_gap"]],0); PCAD=30; C=0.1; USE_CLIN=True
    v=pd.to_numeric(clin["mrs3mo"],errors="coerce"); v=v.where(v!=9)
    yl=pd.Series(np.nan,index=clin.index); yl[v>=3]=1.0; yl[v<=2]=0.0
    head="mRS3mo>=3 deep fusion (DWI gap PCA30 + clinical)"
else:
    DWI=np.concatenate([pp["tr_max"],pp["te_max"]],0); PCAD=50; C=0.1; USE_CLIN=False
    dem=pd.to_numeric(oc["Dementia"],errors="coerce")
    ne3=pd.to_numeric(oc["no_event3m"],errors="coerce"); ne1=pd.to_numeric(oc["no_event1y"],errors="coerce")
    yl=pd.Series(np.nan,index=oc.index); yl[dem==1]=1.0; yl[(dem.isna())&((ne3==1)|(ne1==1))]=0.0
    head="dementia DWI max deep alone"

y=np.array([yl[p] if p in yl.index else np.nan for p in pat])
mask=~np.isnan(y)
idx_all=np.where(mask)[0]
idx_tr=idx_all[idx_all<ntr]        # internal pool (holdout-train with a label)
idx_te=idx_all[idx_all>=ntr]       # final holdout (never used in training/selection)
ytr=y[idx_tr]; yte=y[idx_te]
print(f"[{args.target}] {head}")
print(f"  internal tr={len(idx_tr)} pos={int(ytr.sum())}  |  final holdout te={len(idx_te)} pos={int(yte.sum())}")

def block(X,ti,ei,pca):
    Xtr,Xev=X[ti],X[ei]; cm=np.nanmean(Xtr,0); cm=np.where(np.isnan(cm),0,cm)
    Xtr=np.where(np.isnan(Xtr),cm,Xtr); Xev=np.where(np.isnan(Xev),cm,Xev)
    sc=StandardScaler().fit(Xtr); Xtr,Xev=sc.transform(Xtr),sc.transform(Xev)
    if pca and Xtr.shape[1]>pca:
        pc=PCA(pca,random_state=0).fit(Xtr); Xtr,Xev=pc.transform(Xtr),pc.transform(Xev)
    return Xtr,Xev
def build(ti,ei):
    bt,be=[],[]; a,b=block(DWI,ti,ei,PCAD); bt.append(a); be.append(b)
    if USE_CLIN:
        a,b=block(TAB,ti,ei,None); bt.append(a); be.append(b)
    return np.concatenate(bt,1),np.concatenate(be,1)
def boot(yv,pv,n=3000):
    a=[];ix=np.arange(len(yv))
    for _ in range(n):
        bb=rng.choice(ix,len(ix),True)
        if yv[bb].sum()<2 or (1-yv[bb]).sum()<2: continue
        a.append(roc_auc_score(yv[bb],pv[bb]))
    return np.percentile(a,[2.5,97.5])

# -- internal: 5x5 OOF on holdout-train only --
def oof_tr(yvec,seeds=range(5)):
    o=np.zeros(len(idx_tr)); c=np.zeros(len(idx_tr)); ii=np.arange(len(idx_tr))
    for s in seeds:
        for tr,ev in StratifiedKFold(5,shuffle=True,random_state=s).split(ii,yvec):
            ti,ei=idx_tr[tr],idx_tr[ev]
            Xtr,Xev=build(ti,ei)
            clf=LogisticRegression(C=C,class_weight="balanced",max_iter=2000).fit(Xtr,yvec[tr].astype(int))
            o[ev]+=clf.predict_proba(Xev)[:,1]; c[ev]+=1
    return o/np.maximum(c,1)
oo=oof_tr(ytr); a_in=roc_auc_score(ytr,oo); lo,hi=boot(ytr,oo)
print(f"\n=== internal (5x5 OOF on train only, test fully excluded) ===")
print(f"  OOF AUROC = {a_in:.3f} [{lo:.3f},{hi:.3f}] AP={average_precision_score(ytr,oo):.3f}")

# permutation: is the signal real within train?
perm=np.array([roc_auc_score(yp,oof_tr(yp,seeds=range(2))) for yp in (rng.permutation(ytr) for _ in range(args.nperm))])
pval=(np.sum(perm>=a_in)+1)/(args.nperm+1)
print(f"  permutation: null mean {perm.mean():.3f} max {perm.max():.3f}  p={pval:.4f} -> {'genuine' if pval<0.05 else 'inconclusive'}")

# -- final holdout: train on all of holdout-train -> predict unseen test --
Xtr,Xte=build(idx_tr,idx_te)
clf=LogisticRegression(C=C,class_weight="balanced",max_iter=2000).fit(Xtr,ytr.astype(int))
pte=clf.predict_proba(Xte)[:,1]; a_te=roc_auc_score(yte,pte); lo2,hi2=boot(yte,pte)
print(f"\n=== final holdout (train on all train -> predict unseen test once) ===")
print(f"  holdout AUROC = {a_te:.3f} [{lo2:.3f},{hi2:.3f}] AP={average_precision_score(yte,pte):.3f}")
print(f"\nverdict: internal OOF(train only)={a_in:.3f} (p={pval:.4f}), independent holdout(test)={a_te:.3f}")
