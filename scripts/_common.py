"""Shared CLI helpers for the pipeline scripts (imported after sys.path is set by each script)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def apply_overrides(cfg: dict, sets: list[str] | None) -> dict:
    """--set hparams.reduction=pca fit.max_images=1000 (values parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for s in sets or []:
        key, _, val = s.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def load_model_cfg(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())
