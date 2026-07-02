# configs/

YAML configs for the frozen-probe / feature-extraction pipeline. Loaded by
`shared/config_utils.py`, which supports single-level `_base_` inheritance: a child
config names its parent in `_base_` and is deep-merged on top of it.

| File | Used by |
|---|---|
| `base.yml` | parent for all others — data paths, backbone weights, model dims |
| `singletask_end1_binary.yml` | `extraction/extract_flair_triad.py` |
| `best/4task_single_film_t2frozen_triadpartial.yml` | `shared/linear_probe_diagnostic.py`, `extraction/extract_dwi_embeddings.py` |
| `best/router_slim_v2_frozen.yml` | `preprocessing/skullstrip_dwi.py` |

Fill in the `/path/to/...` placeholders (they mirror `paths.py`). `csv_file` and
`val_csv` are the train / untouched-holdout split CSVs (`pat_id`, `t2_path`, outcome
columns); labels themselves come from `data/csvs/outcomes_final.csv`.

The mRS operating model (`mrs/`) and the `xai/` tools are stand-alone and read paths
from `paths.py`, not from here.
