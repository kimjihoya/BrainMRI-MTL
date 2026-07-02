"""Rank saliency cases by how well the hot region sits on the actual DWI lesion (bright DWI),
then tile the best ones into a montage — quick way to pick publication figures."""
import os,sys,glob,argparse,numpy as np,nibabel as nib,pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion,gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> paths.py
import paths
ap=argparse.ArgumentParser()
ap.add_argument("--dir",default="results/xai/gallery_occ"); ap.add_argument("--suffix",default="_occ")
ap.add_argument("--brain_dir",default=paths.DWI_SKULLSTRIP_DIR)
ap.add_argument("--top",type=int,default=12); ap.add_argument("--pct",type=float,default=92)
ap.add_argument("--out",default="results/xai/gallery_occ/_best_top.png")
a=ap.parse_args()
rows=[]
for cf in sorted(glob.glob(os.path.join(a.dir,f"*{a.suffix}.nii.gz"))):
    pid=os.path.basename(cf).replace(f"{a.suffix}.nii.gz","")
    dwi=nib.load(os.path.join(a.brain_dir,f"{pid}.nii.gz")).get_fdata().astype(np.float32)
    sal=nib.load(cf).get_fdata().astype(np.float32)
    brain=binary_erosion(dwi>0,iterations=2); sal=gaussian_filter(sal,1.2)*brain
    if sal.max()<=0: continue
    sal=sal/sal.max()
    bd=dwi[brain]; thr=np.percentile(bd,a.pct) if bd.size else 0
    lesion=brain&(dwi>thr)                       # bright-DWI = acute lesion proxy
    score=float((sal*lesion).sum()/(sal.sum()+1e-8))   # fraction of saliency mass on lesion
    rows.append((pid,score))
rank=pd.DataFrame(rows,columns=["pid","lesion_overlap"]).sort_values("lesion_overlap",ascending=False)
rank.to_csv(os.path.join(a.dir,"_ranking.csv"),index=False)
top=rank.head(a.top)["pid"].tolist()
print("Top cases (saliency-on-lesion overlap):"); print(rank.head(a.top).to_string(index=False))
# montage: 1 peak slice each
cols=4; nrow=(len(top)+cols-1)//cols
fig,axes=plt.subplots(nrow,cols,figsize=(cols*3.2,nrow*3.2)); axes=np.array(axes).reshape(-1)
for ax in axes: ax.axis("off")
for i,pid in enumerate(top):
    dwi=nib.load(os.path.join(a.brain_dir,f"{pid}.nii.gz")).get_fdata().astype(np.float32)
    sal=nib.load(os.path.join(a.dir,f"{pid}{a.suffix}.nii.gz")).get_fdata().astype(np.float32)
    brain=binary_erosion(dwi>0,iterations=2); sal=gaussian_filter(sal,1.2)*brain; sal=sal/(sal.max()+1e-8)
    z=int(np.argmax(sal.sum((0,1))))
    ax=axes[i]; ax.imshow(np.rot90(dwi[:,:,z]),cmap="gray")
    ax.imshow(np.ma.masked_less(np.rot90(sal[:,:,z]),0.3),cmap="hot",alpha=0.55,vmin=0,vmax=1)
    ax.set_title(f"{pid}  (overlap {rank.set_index('pid').loc[pid,'lesion_overlap']:.2f})",fontsize=9)
fig.suptitle(f"Top {a.top} saliency cases (hot region on DWI lesion)",fontsize=12)
fig.tight_layout(); fig.savefig(a.out,dpi=140,bbox_inches="tight"); print("saved",a.out)
