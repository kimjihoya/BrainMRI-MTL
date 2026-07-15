# Brain MRI Multimodal

Brain MRI Multimodal is a multimodal framework for stroke outcome prediction that combines pretrained MRI foundation models and clinical variables for multimodal stroke outcome prediction.

The primary task is prediction of 3-month functional outcome (modified Rankin Scale ≥3)
from admission DWI and clinical data.

## Highlights

- End-to-end multimodal stroke outcome prediction
- Pretrained MRI foundation models (Triad, BrainIAC, BrainMVP)
- Feature-level fusion of DWI and clinical variables
- Explainability via occlusion, Grad-CAM, SHAP, and modality attribution
- **AUROC:** 0.810 (internal CV), 0.765 (holdout)

## 1. Main task

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

**Model selection**

- Hyperparameters were selected using training-only cross-validation.
- The holdout set was evaluated exactly once.
- An earlier holdout-based selection strategy was discarded to avoid selection bias.

```bash
# train + save the operating checkpoint (3 seeds)
python mrs/train_mrs_endtoend.py --optim adamw --swa 1 --wd 5e-2 --drop 0.6 \
       --unfreeze 1 --seeds 3 --save_final 1 --rep_oof 0.810 --rep_holdout 0.765
# holdout evaluation (run once)
python mrs/train_mrs_endtoend.py --optim adamw --swa 1 --wd 5e-2 --drop 0.6 --seeds 3 --holdout 1
```

## 2. Pipeline

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

### 3. Explainability

The following tools were used to interpret the operating model.

| Tool | Purpose | Output |
|---|---|---|
| `xai/occlusion_saliency.py` | Lesion localization (primary) | occlusion ΔP map: slide a cube, measure the prediction drop. Whole-head input, occlusion restricted to brain |
| `xai/gradcam_dwi.py` | Visual explanation (illustrative) | 3D Grad-CAM heatmap (cleaner on skull-stripped input; see note) |
| `xai/shap_clinical.py` | Clinical feature importance | SHAP beeswarm + mean-\|SHAP\| ranking (clinical effect at an average scan) |
| `xai/modality_attribution.py` | Modality contribution | exact split of each logit into image vs. clinical (the fusion head is linear over the concatenated features) |

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

## 4. Additional tasks

The same backbone, fusion, and validation framework was applied to other stroke outcomes. The same framework was also evaluated on additional stroke prediction tasks.

| Task | Folder | Best AUROC | Note |
|---|---|---|---|
| Early neurological deterioration (END1) | `end1/` | ~0.65 | future event; imaging near-ceiling, frozen probe wins |
| Vascular dementia | `dementia/` | ~0.74 | age-driven; encoder fine-tuning overfits (negative result) |
| MACE (recurrent vascular events) | `mace/` | ~0.69 | driven by cardiac source, not visible on brain MRI |

Imaging helps when the outcome reflects current visible damage (mRS), not a future stochastic event
(END1/MACE) or an age-driven diagnosis (dementia). At ~100–800 patients, a frozen foundation encoder
with a linear/MLP head usually beats end-to-end fine-tuning; mRS is the task where fine-tuning matches
it.

## 5. Installation

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

### Required external resources

| Slot | Used by | Provide |
|---|---|---|
| `paths.py: TRIAD_CKPT` | mRS / dementia encoders | Triad PlainConvUNet-MAE weights (Section 6) |
| `paths.py: BRAINIAC_CKPT` | T2 branch | BrainIAC checkpoint |
| `paths.py: DATA_ROOT` | all image scripts | Directory containing `DWI_preprocessed/`, `FLAIR_preprocessed/`, `T2_preprocessed/`, and `all_data.csv` |
| `train_lightning_multitask.py` | `end1/train_deep_cv.py` | the multitask LightningModule (project-specific) |
| `results/*.npz` caches | frozen probes | precomputed embeddings / scalars (run `extraction/` first) |

## 6. Pretrained foundation models

This project uses publicly available pretrained MRI foundation models, either kept frozen or lightly fine-tuned depending on the experiment. Please cite the original work when using these backbones.

| Backbone | Modality | Role here | Reference |
|---|---|---|---|
| Triad | DWI | Main image encoder (operating mRS model) | [Paper](https://doi.org/10.1016/j.media.2026.103992) · [Code](https://github.com/wangshansong1/Triad) |
| BrainIAC | T2 | Optional T2 branch | [Paper](https://doi.org/10.1038/s41593-026-02202-6) · [Code](https://github.com/AIM-KannLab/BrainIAC) |
| BrainMVP | — | Alternate backbone (optional) | [Paper](https://arxiv.org/abs/2410.10604) · [Code](https://github.com/shaohao011/BrainMVP) |

## 7. Citation

_[TODO: add paper / preprint citation once available.]_
