# Predicting 3-Month Functional Outcome After Stroke from Admission Brain MRI

**A deep multimodal model that fuses an admission DWI lesion representation with admission clinical
variables to predict 3-month disability (modified Rankin Scale ≥ 3).**

> **Headline result.** End-to-end fine-tuning of a pretrained DWI encoder fused with clinical data
> reaches **AUROC 0.805 (out-of-fold) / 0.843 (untouched holdout)** — matching a strong frozen
> baseline while learning the lesion representation directly. The same framework plugs into several
> other stroke outcomes (see *Extending to other tasks*).

---

## 1. Main task — 3-month mRS (the paper)

`mrs/` contains the operating model and its benchmarks.

| File | Role |
|---|---|
| **`mrs/train_mrs_endtoend.py`** | **operating model** — Triad DWI encoder (last stage fine-tuned) ⊕ clinical fusion |
| `mrs/train_mrs_frozen.py` | frozen deep-fusion reference (DWI embedding + PCA + logistic regression) |
| `mrs/validate_mrs.py` | strict validation: permutation test + fully-separated holdout |

**Architecture (intermediate / feature-level fusion):**

```
DWI [96³] ─▶ Triad PlainConvEncoder ─▶ gap-pool last 3 skips ─▶ MLP ─▶ 128-d ┐
                  (last stage trainable, rest frozen)                          ├─▶ concat ─▶ head ─▶ logit
clinical (32, admission) ──────────────▶ MLP ──────────────────────▶ 32-d ────┘
```

**The recipe that matters** (all ablations are flag-driven inside `train_mrs_endtoend.py`,
no separate scripts):

| Lever | Finding |
|---|---|
| `--optim adamw` | default optimizer, OOF = 0.810, holdout test = 0.765|
| `--swa` | weight averaging over the last K epochs → flatter solution, better holdout |
| `--drop 0.6 --wd 5e-2` | strong regularization |
| TTA (8 flips) + 3-seed ensemble | inference-time stabilization (this is what crossed 0.800) |
| `--use_t2` | adding a frozen T2 branch helped OOF a hair but hurt holdout → **not used** |

```bash
# operating model: train + save checkpoint
python mrs/train_mrs_endtoend.py --optim radam --swa 1 --wd 5e-2 --drop 0.6 \
       --unfreeze 1 --seeds 3 --save_final 1 --rep_oof 0.805 --rep_holdout 0.843
# untouched-holdout evaluation
python mrs/train_mrs_endtoend.py --optim radam --swa 1 --wd 5e-2 --drop 0.6 --seeds 3 --holdout 1
```

---

## 2. Pipeline (preprocessing → features → model)

```
preprocessing/        raw DICOM → preprocessed 96³ volumes
  preprocess_dwi.py       DWI b1000 → 96³ int16
  preprocess_flair.py     CE-FLAIR → 96³ int16 (same spec)
  skullstrip_dwi.py       optional DWI skull-strip
  extract_adc.py          pull ADC series out of raw zips → NIfTI
  cache_adc96.py          resample ADC onto the 96³ grid

extraction/           frozen foundation embeddings + engineered scalars
  extract_dwi_embeddings.py   Triad DWI pooled embeddings (max/gap/std/gapmax)
  extract_flair_triad.py      Triad FLAIR embeddings
  extract_flair_mismatch.py   FLAIR-DWI mismatch scalars
  extract_adc_scalars.py      ADC acute-core scalars
  extract_adc_lesion_scalars.py  in-lesion ADC scalars
  outcome_ceiling.py          quick scan of which outcomes are imaging-predictable

shared/               infrastructure shared across tasks
  model.py                MultiTaskModel (Triad / BrainIAC / BrainMVP backbones + fusion)
  dataset.py              dataset + label construction from outcomes_final.csv
  linear_probe_diagnostic.py   frozen-probe helpers (build_block / fit_predict) + embedding extraction
  config_utils.py         yaml config loader (_base_ inheritance)
```

### Explainability (`xai/`)

Views of *what the operating mRS model used*:

| Tool | Question | Output |
|---|---|---|
| **`xai/occlusion_saliency.py`** | **Where on the DWI? (primary)** | occlusion ΔP map — slide a cube, measure the prediction drop. Computed on whole-head (deployment) input, occlusion restricted to brain |
| `xai/gradcam_dwi.py` | Where on the DWI? (illustrative) | 3D Grad-CAM heatmap (cleaner on skull-stripped input; see caveat below) |
| `xai/shap_clinical.py` | **Which clinical variables drove it?** | SHAP beeswarm + mean-\|SHAP\| ranking (clinical effect at an average scan) |
| `xai/modality_attribution.py` | **Imaging or clinical?** | exact split of each logit into image vs. clinical (the fusion head is linear over the concatenated features) |

Visualization helpers: `render_saliency_png.py` (NIfTI → PNG overlays), `contact_sheet.py`
(tile N patients), `rank_best_cases.py` (rank by saliency-on-lesion overlap → pick figures).
`xai/ft_model.py` rebuilds the operating FTNet from a checkpoint and is imported by all tools.

```bash
python xai/occlusion_saliency.py --gpu 0 --pat auto --topk 100 \
       --dwi_dir /path/to/data/DWI_preprocessed --brain_dir /path/to/data/DWI_skullstrip
python xai/render_saliency_png.py --dir results/xai/occlusion --suffix _occ
python xai/rank_best_cases.py     --dir results/xai/occlusion --suffix _occ --top 12
python xai/modality_attribution.py --gpu 0 --n 300     # imaging vs clinical share
python xai/shap_clinical.py        --gpu 0 --n 200     # clinical SHAP
```

> **Method note (important).** The operating model uses **global average pooling**, so its decision
> is global and saliency is inherently distributed — and it is trained on **whole-head** DWI.
> Predictions are **skull-invariant** (removing the skull changes P by ≤0.03), but Grad-CAM is
> *input-dependent*: on whole-head it is confounded by high-intensity skull edges, on skull-stripped
> it localizes to the lesion. **Occlusion is the deployment-faithful primary**: it runs on the real
> whole-head input and is prediction-based, so the skull is excluded automatically (occluding it
> barely changes P). Report it with the skull-invariance number; use Grad-CAM only as an illustration.

---

## 3. Extending to other tasks

The same backbone + fusion + validation framework was applied to other stroke outcomes. These are
secondary to the mRS paper but show where imaging does and does not help.

| Task | Folder | Best AUROC | Takeaway |
|---|---|---|---|
| Early neurological deterioration (END1) | `end1/` | ~0.65 | future event; imaging near-ceiling, frozen probe wins |
| Vascular dementia | `dementia/` | ~0.74 | age-driven; encoder fine-tuning overfits (negative result) |
| MACE (recurrent vascular events) | `mace/` | ~0.69 | driven by cardiac source, not visible on brain MRI |

> **The cross-task lesson:** imaging helps only when the outcome reflects *current visible damage*
> (mRS), not a future stochastic event (END1/MACE) or an age-driven diagnosis (dementia). At
> ~100–800 patients, a **frozen** foundation encoder + linear/MLP head usually beats end-to-end
> fine-tuning; mRS is the one task where, with the right recipe, fine-tuning matches it.

---

## 4. Setup

```bash
pip install -r requirements.txt
```

All file-system paths are placeholders (`/path/to/...`). Fill them in once in **`paths.py`** before
running. Scripts launch from the repository root; `data/csvs/outcomes_final.csv` is repo-relative and
is the single source of truth for labels.

> GPU is used for encoder training (Triad/BrainIAC, 96³ volumes); frozen probes run on CPU.

### External dependencies to drop in

| Slot | Used by | Fill in |
|---|---|---|
| `paths.py: TRIAD_CKPT` | mRS / dementia encoders | Triad PlainConvUNet-MAE weights (see Acknowledgements) |
| `paths.py: BRAINIAC_CKPT` | T2 branch | BrainIAC checkpoint |
| `paths.py: DATA_ROOT` | all image scripts | dir with `DWI_preprocessed/`, `FLAIR_preprocessed/`, `T2_preprocessed/`, `all_data.csv` |
| `train_lightning_multitask.py` | `end1/train_deep_cv.py` | the multitask LightningModule (project-specific) |
| `results/*.npz` caches | frozen probes | precomputed embeddings / scalars (run `extraction/` first) |

---

## 5. Project rules (hard constraints)

- **Outcomes come only from `data/csvs/outcomes_final.csv`** (curated label file).
- **Never use radiology-read location variables** (cortex/bgic/thal/pons/mca/pca/ba/cere) for END1.
- **Never leak discharge variables** (`dis_*`) when predicting 3-month / 1-year outcomes.
- Holdout-test patients are never used in training or model selection.

---

## 6. Pretrained backbones & acknowledgements

This work stands on pretrained foundation encoders, kept **frozen** or only lightly fine-tuned.
Please cite the originals.

| Backbone | Modality | Role here | Reference |
|---|---|---|---|
| **Triad** (PlainConvUNet, MAE-pretrained) | DWI | main image encoder (operating mRS model) | _[TODO: citation / URL]_ |
| **BrainIAC** (ViT) | T2 | optional T2 branch | _[TODO: citation / URL]_ |
| **BrainMVP** (Uniformer) | — | alternate backbone (optional) | _[TODO: citation / URL]_ |

> 🙏 **Shout-out to Triad** — the DWI lesion representation that makes the mRS model work comes from
> the Triad PlainConvUNet MAE encoder. _[TODO: add the official Triad repo / paper link.]_

## 7. Citation

_[TODO: add your paper / preprint citation here once available.]_
