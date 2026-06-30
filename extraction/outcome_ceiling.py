"""Quick ceiling scan for new targets that reflect 'currently visible' damage (dis_mrs / mrs1y).
Same pipeline as the mRS deep fusion: frozen DWI(Triad) embedding + clinical tab.
Compares clin-only / DWI-deep / DWI+clin fusion OOF. No leakage (PCA/scaler fit inside each fold)."""
import numpy as np, pandas as pd, warnings, argparse
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

ap=argparse.ArgumentParser()
ap.add_argument("--target",default="dis_mrs"); ap.add_argument("--thr",type=float,default=3)
args=ap.parse_args()
rng=np.random.default_rng(0)
lp=np.load("results/linear_probe_features.npz",allow_pickle=True)
pp=np.load("results/pooling_dwi_features.npz",allow_pickle=True)
DWI=np.concatenate([pp["tr_gap"],pp["te_gap"]],0)      # gap pooling (best for mRS)
TAB=np.concatenate([lp["tr_tab"],lp["te_tab"]],0)
pat=np.concatenate([lp["tr_pat_id"],lp["te_pat_id"]]).astype(str)

oc=pd.read_csv("data/csvs/outcomes_final.csv",dtype={"record_id":str},encoding="utf-8-sig")
oc.columns=[c.strip().replace("﻿","") for c in oc.columns]; oc=oc.set_index("record_id")
v=pd.to_numeric(oc[args.target],errors="coerce"); v=v.where(v!=9)   # drop 9 = unknown
yl=pd.Series(np.nan,index=oc.index); yl[v>=args.thr]=1.0; yl[v<args.thr]=0.0
y=np.array([yl[p] if p in yl.index else np.nan for p in pat])
mask=~np.isnan(y); idx=np.where(mask)[0]; ysub=y[idx]
print(f"target={args.target}>={args.thr}  cohort={int(mask.sum())} pos={int(ysub.sum())} ({ysub.mean():.1%})")

def ci(yv,pv,n=3000):
    a=[];ix=np.arange(len(yv))
    for _ in range(n):
        b=rng.choice(ix,len(ix),True)
        if yv[b].sum()<2 or (1-yv[b]).sum()<2: continue
        a.append(roc_auc_score(yv[b],pv[b]))
    return np.percentile(a,[2.5,97.5])
def block(X,ti,ei,pca):
    Xtr,Xev=X[ti],X[ei]; cm=np.nanmean(Xtr,0); cm=np.where(np.isnan(cm),0,cm)
    Xtr=np.where(np.isnan(Xtr),cm,Xtr); Xev=np.where(np.isnan(Xev),cm,Xev)
    sc=StandardScaler().fit(Xtr); Xtr,Xev=sc.transform(Xtr),sc.transform(Xev)
    if pca and Xtr.shape[1]>pca:
        pc=PCA(pca,random_state=0).fit(Xtr); Xtr,Xev=pc.transform(Xtr),pc.transform(Xev)
    return Xtr,Xev
def oof(mode,C=0.1,seeds=range(5),nf=5):
    o=np.zeros(len(idx)); c=np.zeros(len(idx)); ii=np.arange(len(idx))
    for s in seeds:
        for tr,ev in StratifiedKFold(nf,shuffle=True,random_state=s).split(ii,ysub):
            ti,ei=idx[tr],idx[ev]; blk=[]
            if mode in ("clin","fusion"):
                tt,te=block(TAB,ti,ei,None); blk.append((tt,te))
            if mode in ("dwi","fusion"):
                dt,de=block(DWI,ti,ei,30); blk.append((dt,de))
            Xtr=np.concatenate([b[0] for b in blk],1); Xev=np.concatenate([b[1] for b in blk],1)
            clf=LogisticRegression(C=C,class_weight="balanced",max_iter=3000).fit(Xtr,y[ti].astype(int))
            o[ev]+=clf.predict_proba(Xev)[:,1]; c[ev]+=1
    return o/np.maximum(c,1)
def rep(name,mode):
    best=(-1,None)
    for C in [0.05,0.1,0.3,1.0]:
        p=oof(mode,C); a=roc_auc_score(ysub,p)
        if a>best[0]: best=(a,p)
    a,p=best; lo,hi=ci(ysub,p); apv=average_precision_score(ysub,p)
    star=" ★0.8+" if a>=0.8 else ""
    print(f"  {name:22s} {a:.3f} [{lo:.3f},{hi:.3f}] AP={apv:.3f}{star}")
    return a,p

print("same-cohort comparison:")
rep("clin only", "clin")
rep("DWI deep only", "dwi")
rep("DWI+clin fusion", "fusion")
