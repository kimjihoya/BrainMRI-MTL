"""Render Grad-CAM NIfTI outputs as PNG overlays (no NIfTI viewer needed).

For each {pid}_dwi.nii.gz + {pid}_gradcam.nii.gz pair, overlays the Grad-CAM heatmap (hot colormap)
on the DWI in grayscale, at axial slices through the strongest activation, and saves a montage PNG.
"""
import os, glob, argparse, numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/xai/gradcam")
ap.add_argument("--suffix", default="_gradcam", help="saliency file suffix, e.g. _gradcam or _occ")
ap.add_argument("--n_slices", type=int, default=6)
ap.add_argument("--alpha", type=float, default=0.45)
ap.add_argument("--erode", type=int, default=3, help="brain-mask erosion iters (drops the dura/edge rim)")
ap.add_argument("--smooth", type=float, default=1.5, help="gaussian sigma for the heatmap (0=off)")
ap.add_argument("--thresh", type=float, default=0.25, help="hide saliency below this fraction of max")
args = ap.parse_args()

cams = sorted(glob.glob(os.path.join(args.dir, f"*{args.suffix}.nii.gz")))
print(f"rendering {len(cams)} cases -> {args.dir}/*.png")
for cf in cams:
    pid = os.path.basename(cf).replace(f"{args.suffix}.nii.gz", "")
    dwi = nib.load(os.path.join(args.dir, f"{pid}_dwi.nii.gz")).get_fdata().astype(np.float32)
    cam = nib.load(cf).get_fdata().astype(np.float32)
    # restrict saliency to brain tissue (DWI>0), eroded to drop the dura/skull rim
    from scipy.ndimage import binary_erosion, gaussian_filter
    brain = binary_erosion(dwi > 0, iterations=args.erode)
    cam = cam * brain
    if args.smooth > 0:
        cam = gaussian_filter(cam, sigma=args.smooth) * brain   # smooth, then re-mask
    cam = cam / (cam.max() + 1e-8)
    # pick axial slices with the most in-brain Grad-CAM signal
    z_energy = cam.sum(axis=(0, 1))
    if z_energy.sum() == 0:
        zs = np.linspace(dwi.shape[2] * 0.3, dwi.shape[2] * 0.7, args.n_slices).astype(int)
    else:
        top = np.argsort(z_energy)[::-1][:args.n_slices]
        zs = np.sort(top)
    dnorm = (dwi - dwi.min()) / (dwi.max() - dwi.min() + 1e-8)
    cols = args.n_slices
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.2))
    if cols == 1: axes = [axes]
    for ax, z in zip(axes, zs):
        bg = np.rot90(dnorm[:, :, z]); hm = np.rot90(cam[:, :, z])
        ax.imshow(bg, cmap="gray", interpolation="nearest")
        ax.imshow(np.ma.masked_less(hm, args.thresh), cmap="hot", alpha=args.alpha,
                  interpolation="bilinear", vmin=0, vmax=1)
        ax.set_title(f"z={z}", fontsize=9); ax.axis("off")
    fig.suptitle(f"{pid}  —  saliency on DWI (hot = model focus)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(args.dir, f"{pid}_overlay.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  {pid} -> {pid}_overlay.png")
print("done.")
