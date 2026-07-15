"""Tile per-patient Grad-CAM peak slices into a single contact sheet for quick browsing."""
import os,re,glob,argparse,numpy as np,nibabel as nib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, gaussian_filter
ap=argparse.ArgumentParser()
ap.add_argument("--dir",default="results/xai/gallery"); ap.add_argument("--cols",type=int,default=10)
ap.add_argument("--out",default="results/xai/gallery/_contact_sheet.png")
ap.add_argument("--title",default="Grad-CAM contact sheet")
ap.add_argument("--suffix",default="_gradcam")
a=ap.parse_args()
cams=sorted(glob.glob(os.path.join(a.dir,f"*{a.suffix}.nii.gz")))
n=len(cams); rows=(n+a.cols-1)//a.cols
fig,axes=plt.subplots(rows,a.cols,figsize=(a.cols*1.6,rows*1.6))
axes=np.array(axes).reshape(-1)
for ax in axes: ax.axis("off")
for i,cf in enumerate(cams):
    pid=os.path.basename(cf).replace(f"{a.suffix}.nii.gz","")
    dwi=nib.load(os.path.join(a.dir,f"{pid}_dwi.nii.gz")).get_fdata().astype(np.float32)
    cam=nib.load(cf).get_fdata().astype(np.float32)
    brain=binary_erosion(dwi>0,iterations=3); cam=gaussian_filter(cam,1.8)*brain
    cam=cam/(cam.max()+1e-8)
    z=int(np.argmax(cam.sum((0,1))))
    bg=np.rot90((dwi-dwi.min())/(dwi.max()-dwi.min()+1e-8)[...,z] if False else np.rot90((dwi[:,:,z]-dwi.min())/(dwi.max()-dwi.min()+1e-8)))
    ax=axes[i]; ax.imshow(np.rot90(dwi[:,:,z]),cmap="gray")
    ax.imshow(np.ma.masked_less(np.rot90(cam[:,:,z]),0.3),cmap="hot",alpha=0.55,vmin=0,vmax=1)
    ax.set_title(re.sub(r"^\d{6}-","",pid),fontsize=6,pad=1)
fig.suptitle(f"{a.title} — {n} patients",fontsize=11)
fig.tight_layout(); fig.savefig(a.out,dpi=140,bbox_inches="tight"); print(f"saved {a.out}")
