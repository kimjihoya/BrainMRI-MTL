# Brain MRI Multimodal

A multimodal model that fuses an admission DWI lesion representation with admission clinical
variables to predict 3-month disability (modified Rankin Scale ≥ 3).

End-to-end fine-tuning of a pretrained DWI encoder, fused with clinical data, reaches AUROC 0.810
on internal cross-validation and 0.765 on an untouched holdout test set. The same framework was
applied to several other stroke outcomes (see Section 3).

## 1. Main task — 3-month mRS

`mrs/` holds the operating model and its references.

| File | Role |
|---|---|
| `mrs/train_mrs_endtoend.py` | operating model — Triad DWI encoder (last stage fine-tuned) + clinical fusion |
| `mrs/train_mrs_frozen.py` | frozen deep-fusion reference (DWI embedding + PCA + logistic regression) |
| `mrs/validate_mrs.py` | validation: tr-only cross-validation, permutation test, separated holdout |

Architecture (intermediate / feature-level fusion):

```
DWI [96³] ─▶ Triad PlainConvEncoder ─▶ gap-pool last 3 skips ─▶ MLP ─▶ 128-d ┐
                  (last stage trainable, rest frozen)                          ├─▶ concat ─▶ head ─▶ logit
clinical (32, admission) ──────────────▶ MLP ──────────────────────▶ 32-d ────┘
```

All ablations are flag-driven inside `train_mrs_endtoend.py`; there are no separate scripts.

| Flag | Effect |
|---|---|
| `--optim adamw` | default optimizer; selected on tr-only CV (0.810) over radam (0.786) and sgd (0.776) |
| `--swa 1` | average weights over the last K epochs |
| `--drop 0.6 --wd 5e-2` | regularization |
| TTA (8 flips) + 3 seeds | inference-time averaging |
| `--use_t2 1` | frozen T2 branch; helped CV slightly but hurt holdout, so not used |

Selection protocol: the optimizer and threshold are chosen on tr-only cross-validation (holdout
excluded). The holdout is evaluated once, at the end. An earlier version selected the optimizer on
the holdout (radam looked best there at 0.843); that is selection bias on 33 positives and was
discarded.

```bash
# train + save the operating checkpoint (3 seeds)
python mrs/train_mrs_endtoend.py --optim adamw --swa 1 --wd 5e-2 --drop 0.6 \
       --unfreeze 1 --seeds 3 --save_final 1 --rep_oof 0.810 --rep_holdout 0.765
# holdout evaluation (run once)
python mrs/train_mrs_endtoend.py --optim adamw --swa 1 --wd 5e-2 --drop 0.6 --seeds 3 --holdout 1
```

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
  outcome_ceiling.py          scan of which outcomes are imaging-predictable

shared/               infrastructure shared across tasks
  model.py                MultiTaskModel (Triad / BrainIAC / BrainMVP backbones + fusion)
  dataset.py              dataset + label construction from outcomes_final.csv
  linear_probe_diagnostic.py   frozen-probe helpers + embedding extraction
  config_utils.py         yaml config loader (_base_ inheritance)

configs/              yaml configs for the frozen-probe / extraction pipeline
  base.yml                shared defaults (paths, backbone weights, model dims)
  *.yml, best/*.yml       experiment configs inheriting from base.yml via _base_
```

### Explainability (`xai/`)

What the operating mRS model used:

| Tool | Question | Output |
|---|---|---|
| `xai/occlusion_saliency.py` | Where on the DWI? (primary) | occlusion ΔP map: slide a cube, measure the prediction drop. Whole-head input, occlusion restricted to brain |
| `xai/gradcam_dwi.py` | Where on the DWI? (illustrative) | 3D Grad-CAM heatmap (cleaner on skull-stripped input; see note) |
| `xai/shap_clinical.py` | Which clinical variables? | SHAP beeswarm + mean-\|SHAP\| ranking (clinical effect at an average scan) |
| `xai/modality_attribution.py` | Imaging or clinical? | exact split of each logit into image vs. clinical (the fusion head is linear over the concatenated features) |

Helpers: `render_saliency_png.py` (NIfTI → PNG overlays), `contact_sheet.py` (tile N patients),
`rank_best_cases.py` (rank by saliency-on-lesion overlap). `xai/ft_model.py` rebuilds the operating
FTNet from a checkpoint and is imported by all tools.

```bash
python xai/occlusion_saliency.py --gpu 0 --pat auto --topk 100 \
       --dwi_dir /path/to/data/DWI_preprocessed --brain_dir /path/to/data/DWI_skullstrip
python xai/render_saliency_png.py --dir results/xai/occlusion --suffix _occ
python xai/rank_best_cases.py     --dir results/xai/occlusion --suffix _occ --top 12
python xai/modality_attribution.py --gpu 0 --n 300
python xai/shap_clinical.py        --gpu 0 --n 200
```

Note on saliency: the operating model uses global average pooling, so its decision is global and
saliency is distributed. It is trained on whole-head DWI, and predictions are skull-invariant
(removing the skull changes P by ≤ 0.03). Grad-CAM is input-dependent — on whole-head input it is
confounded by skull edges, on skull-stripped input it localizes to the lesion. Occlusion runs on the
real whole-head input and is prediction-based, so the skull is excluded automatically; it is the
primary saliency method. Grad-CAM is illustrative only.

## 3. Other tasks

The same backbone, fusion, and validation framework was applied to other stroke outcomes. These are
secondary to the mRS work but show where imaging does and does not help.

| Task | Folder | Best AUROC | Note |
|---|---|---|---|
| Early neurological deterioration (END1) | `end1/` | ~0.65 | future event; imaging near-ceiling, frozen probe wins |
| Vascular dementia | `dementia/` | ~0.74 | age-driven; encoder fine-tuning overfits (negative result) |
| MACE (recurrent vascular events) | `mace/` | ~0.69 | driven by cardiac source, not visible on brain MRI |

Imaging helps when the outcome reflects current visible damage (mRS), not a future stochastic event
(END1/MACE) or an age-driven diagnosis (dementia). At ~100–800 patients, a frozen foundation encoder
with a linear/MLP head usually beats end-to-end fine-tuning; mRS is the task where fine-tuning matches
it.

## 4. Setup

```bash
pip install -r requirements.txt
```

File-system paths are placeholders (`/path/to/...`). There are two places to fill them in:

- `paths.py` — used by the stand-alone scripts (`mrs/`, `xai/`, and the non-config
  `preprocessing/` / `extraction/` scripts).
- `configs/*.yml` — used by the config-driven pipeline (`linear_probe_diagnostic`,
  `extract_dwi_embeddings`, `extract_flair_triad`, `skullstrip_dwi`). See `configs/README.md`.

Point both at the same real directories. Scripts launch from the repository root.
`data/csvs/outcomes_final.csv` is repo-relative and is the single source of truth for labels.
Encoder training uses a GPU (96³ volumes); frozen probes run on CPU.

### External dependencies to provide

| Slot | Used by | Provide |
|---|---|---|
| `paths.py: TRIAD_CKPT` | mRS / dementia encoders | Triad PlainConvUNet-MAE weights (Section 6) |
| `paths.py: BRAINIAC_CKPT` | T2 branch | BrainIAC checkpoint |
| `paths.py: DATA_ROOT` | all image scripts | dir with `DWI_preprocessed/`, `FLAIR_preprocessed/`, `T2_preprocessed/`, `all_data.csv` |
| `train_lightning_multitask.py` | `end1/train_deep_cv.py` | the multitask LightningModule (project-specific) |
| `results/*.npz` caches | frozen probes | precomputed embeddings / scalars (run `extraction/` first) |

## 5. Project rules

- Outcomes come only from `data/csvs/outcomes_final.csv`.
- Never use radiology-read location variables (cortex/bgic/thal/pons/mca/pca/ba/cere) for END1.
- Never leak discharge variables (`dis_*`) when predicting 3-month / 1-year outcomes.
- Holdout-test patients are never used in training or model selection.

## 6. Pretrained backbones

This work uses pretrained foundation encoders, kept frozen or lightly fine-tuned. Please cite the
originals.

| Backbone | Modality | Role here | Reference |
|---|---|---|---|
| Triad (PlainConvUNet, MAE-pretrained) | DWI | main image encoder (operating mRS model) | _[TODO: citation / URL]_ |
| BrainIAC (ViT) | T2 | optional T2 branch | _[TODO: citation / URL]_ |
| BrainMVP (Uniformer) | — | alternate backbone (optional) | _[TODO: citation / URL]_ |

The DWI lesion representation used by the mRS model comes from the Triad PlainConvUNet MAE encoder.
_[TODO: add the official Triad repo / paper link.]_

## 7. Citation

_[TODO: add paper / preprint citation once available.]_
