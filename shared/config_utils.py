"""
Config loader with _base_ inheritance.

Usage:
    config = load_config("configs/main_gated.yml")

If an experiment yml sets the _base_ key to a parent config,
the child config is deep-merged on top of the parent.
"""

import os
import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str) -> dict:
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    base_key = cfg.pop("_base_", None)
    if base_key:
        base_path = os.path.join(os.path.dirname(path), base_key)
        base_cfg  = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def save_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
