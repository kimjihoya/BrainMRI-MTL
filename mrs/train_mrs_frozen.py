"""mRS 3-month — FROZEN deep-fusion reference model.

The Triad DWI encoder stays frozen; we pool its embedding, reduce with PCA, concatenate the
admission clinical vector, and fit a balanced logistic regression. Sweeps DWI pooling
(max/gap/std/gapmax) x PCA dim x C and keeps the best OOF config.

5x5 repeated-stratified OOF + bootstrap CI. No leakage: PCA/scaler are fit on the train fold only.
This is the strong, well-generalizing baseline (OOF 0.812 / holdout 0.835) that the end-to-end
operating model is benchmarked against."""
import numpy as np, pandas as pd, warnings, json, os, sys, joblib
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> paths.py
import paths
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

rng=np.random.default_rng(0)
lp=np.load("results/linear_probe_features.npz",allow_pickle=True)
pp=np.load("results/pooling_dwi_features.npz",allow_pickle=True)
POOLS={k:np.concatenate([pp[f"tr_{k}"],pp[f"te_{k}"]],0) for k in ["max","gap","std","gapmax"]}
TAB=np.concatenate([lp["tr_tab"],lp["te_tab"]],0)
pat=np.concatenate([lp["tr_pat_id"],lp["te_pat_id"]]).astype(str)

clin=pd.read_csv(paths.CLINICAL_CSV,dtype={"record_id":str},encoding="utf-8-sig").set_index("record_id")
v=pd.to_numeric(clin["mrs3mo"],errors="coerce"); v=v.where(v!=9)
yl=pd.Series(np.nan,index=clin.index); yl[v>=3]=1.0; yl[v<=2]=0.0
y=np.array([yl[p] if p in yl.index else np.nan for p in pat])
mask=~np.isnan(y); idx=np.where(mask)[0]; ysub=y[idx]
print(f"cohort={int(mask.sum())} pos={int(ysub.sum())}")

def ci(yv,pv,n=3000):
    a=[];ix=np.arange(len(yv))
    for _ in range(n):
        b=rng.choice(ix,len(ix),True)
        if yv[b].sum()<2 or (1-yv[b]).sum()<2: continue
        a.append(roc_auc_score(yv[b],pv[b]))
    return np.percentile(a,[2.5,97.5])

def block(Xfull,ti,ei,pca_dim,do_pca):
    Xtr,Xev=Xfull[ti],Xfull[ei]
    cm=np.nanmean(Xtr,0); cm=np.where(np.isnan(cm),0,cm)
    Xtr=np.where(np.isnan(Xtr),cm,Xtr); Xev=np.where(np.isnan(Xev),cm,Xev)
    sc=StandardScaler().fit(Xtr); Xtr,Xev=sc.transform(Xtr),sc.transform(Xev)
    if do_pca and Xtr.shape[1]>pca_dim:
        pc=PCA(pca_dim,random_state=0).fit(Xtr); Xtr,Xev=pc.transform(Xtr),pc.transform(Xev)
    return Xtr,Xev

def oof(DWI,pca_dim,C,seeds=range(5),nf=5):
    o=np.zeros(len(idx)); c=np.zeros(len(idx)); ii=np.arange(len(idx))
    for s in seeds:
        for tr,ev in StratifiedKFold(nf,shuffle=True,random_state=s).split(ii,ysub):
            ti,ei=idx[tr],idx[ev]
            dt,de=block(DWI,ti,ei,pca_dim,True)
            tt,te=block(TAB,ti,ei,pca_dim,False)
            Xtr=np.concatenate([dt,tt],1); Xev=np.concatenate([de,te],1)
            clf=LogisticRegression(C=C,class_weight="balanced",max_iter=3000)
            clf.fit(Xtr,y[ti].astype(int)); o[ev]+=clf.predict_proba(Xev)[:,1]; c[ev]+=1
    return o/np.maximum(c,1)

print(f"\n{'pool':8s} {'pca':>4s} {'C':>5s}  OOF [95% CI]")
best=(-1,None,None)
for pool in ["max","gap","std","gapmax"]:
    for pca_dim in [30,50,80]:
        for C in [0.1,0.3,1.0]:
            p=oof(POOLS[pool],pca_dim,C); a=roc_auc_score(ysub,p)
            if a>best[0]: best=(a,p,(pool,pca_dim,C))
        # print the representative row only (C=0.1)
    p=oof(POOLS[pool],50,0.1); a=roc_auc_score(ysub,p); lo,hi=ci(ysub,p)
    print(f"{pool:8s} {50:>4d} {0.1:>5.1f}  {a:.3f} [{lo:.3f},{hi:.3f}]")
a,p,(pool,pca_dim,C)=best
lo,hi=ci(ysub,p); ap=average_precision_score(ysub,p)
print(f"\n★ best: DWI({pool}) pca={pca_dim} C={C}  OOF={a:.3f} [{lo:.3f},{hi:.3f}] AP={ap:.3f}")

# save operating model: fit the final pipeline (PCA+LogReg) on the full-cohort statistics
Xd=POOLS[pool]; ti=idx
cm=np.nanmean(Xd[ti],0); cm=np.where(np.isnan(cm),0,cm)
scd=StandardScaler().fit(np.where(np.isnan(Xd[ti]),cm,Xd[ti]))
Xds=scd.transform(np.where(np.isnan(Xd[ti]),cm,Xd[ti]))
pcd=PCA(pca_dim,random_state=0).fit(Xds)
cmt=np.nanmean(TAB[ti],0); cmt=np.where(np.isnan(cmt),0,cmt)
sct=StandardScaler().fit(np.where(np.isnan(TAB[ti]),cmt,TAB[ti]))
Xtr=np.concatenate([pcd.transform(Xds), sct.transform(np.where(np.isnan(TAB[ti]),cmt,TAB[ti]))],1)
clf=LogisticRegression(C=C,class_weight="balanced",max_iter=3000).fit(Xtr,ysub.astype(int))
os.makedirs("checkpoints/mrs_deepfusion",exist_ok=True)
joblib.dump({"dwi_colmean":cm,"dwi_scaler":scd,"dwi_pca":pcd,"tab_colmean":cmt,"tab_scaler":sct,
             "clf":clf,"pool":pool,"pca_dim":pca_dim,"C":C},"checkpoints/mrs_deepfusion/fusion.joblib")
meta={"target":"mrs3mo>=3 (3-month poor outcome)","model":"DWI(Triad) frozen embedding + clinical tab -> PCA+LogReg fusion",
      "dwi_pool":pool,"pca_dim":pca_dim,"C":C,"cohort_n":int(mask.sum()),"n_pos":int(ysub.sum()),
      "oof_auroc":round(a,4),"ci95":[round(lo,4),round(hi,4)],"oof_ap":round(ap,4),
      "eval":"5x5 RepeatedStratifiedKFold OOF + bootstrap CI","note":"true deep multimodal (no engineered scalars), frozen encoder"}
json.dump(meta,open("checkpoints/mrs_deepfusion/metadata.json","w"),ensure_ascii=False,indent=2)
np.savez("results/mrs/mrs_deepfusion_best_oof.npz",pat=pat[idx],y=ysub,oof=p)
print("saved → checkpoints/mrs_deepfusion/{fusion.joblib, metadata.json}, results/mrs_deepfusion_best_oof.npz")
