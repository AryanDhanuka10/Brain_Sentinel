"""Deterministic preprocessing: BMAD RGB PNG -> recovered 8-bit index -> normalised float array.

Nothing here is fitted on data. Train, valid and test go through exactly the same code.
No resizing, no random ops, no histogram equalisation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from .palette import Palette, brain_mask, corner_bg, load_rgb

REPO = Path(__file__).resolve().parents[3]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_cfg(cfg: dict) -> dict:
    """Copy of the data config with `root` absolute (relative paths are relative to the repo root)."""
    c = dict(cfg)
    root = Path(c["root"])
    c["root"] = str(root if root.is_absolute() else REPO / root)
    return c


def load_pp_config(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or REPO / "configs" / "preprocess.yaml").read_text())


def file_list_sha256(paths) -> str:
    """Hash of the sorted file NAMES; proves which files a statistic was fitted on."""
    names = sorted(Path(p).name for p in paths)
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def get_norm(pp: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(mean, std) per channel. 'imagenet' is a constant; 'train_stats' must come from train/good."""
    mode = pp["normalize"]["mode"]
    if mode == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if mode == "train_stats":
        path = Path(pp["normalize"]["train_stats_path"])
        path = path if path.is_absolute() else REPO / path
        d = json.loads(path.read_text())
        if d.get("fitted_on") != "train/good":
            raise ValueError(f"{path}: not fitted on train/good (fitted_on={d.get('fitted_on')!r})")
        return (float(d["mean"]),) * 3, (float(d["std"]),) * 3
    raise ValueError(f"unknown normalize.mode: {mode!r}")


def load_index_image(path: str | Path, pal: Palette, size: int = 240) -> tuple[np.ndarray, np.ndarray, int]:
    """PNG -> (idx, brain, bg).

    idx   (H, W) uint8: palette index with THIS image's own corner background subtracted (clip at 0).
    brain (H, W) bool : palette.brain_mask (HYPOTHESIS, metadata only; never used to zero pixels here).
    bg    int         : this image's corner background (0 for most images, 1..10 for 33 known ones).
    Raises on wrong size, colours outside the palette, non-uniform corners, or background > tolerance.
    """
    rgb = load_rgb(Path(path))
    if rgb.shape[:2] != (size, size):
        raise ValueError(f"{path}: expected {size}x{size}, got {rgb.shape[:2]}")
    idx = pal.encode(rgb)
    brain = brain_mask(idx)
    bg = corner_bg(idx)
    out = np.clip(idx.astype(np.int16) - bg, 0, 255).astype(np.uint8)
    return out, brain, bg


def to_network_input(idx: np.ndarray, mean, std) -> np.ndarray:
    """(H, W) uint8 index -> (3, H, W) float32: index/255, gray replicated x3, then (x - mean) / std."""
    if idx.dtype != np.uint8 or idx.ndim != 2:
        raise ValueError(f"expected 2-D uint8 index, got {idx.dtype} {idx.shape}")
    std_a = np.asarray(std, dtype=np.float32)
    if not (std_a > 0).all():
        raise ValueError(f"std must be > 0, got {std}")
    x = np.repeat((idx.astype(np.float32) / 255.0)[None], 3, axis=0)
    out = (x - np.asarray(mean, dtype=np.float32)[:, None, None]) / std_a[:, None, None]
    out = np.ascontiguousarray(out, dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError("non-finite values after normalisation")
    return out
