"""Builds a TINY dataset in the BMAD Brain on-disk layout inside a pytest tmp dir.

NON-MEDICAL: a smooth-textured disc ('brain') on a zero background with an optional bright blob
('lesion'), rendered through a 256-colour monotone palette. Purpose: exercise the real loaders, index
builder, training/evaluation scripts and the guard rails end to end. It is never written under data/ and
no metric computed from it is ever reported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from brainsentinel.data.index import add_flags, build_rows, columns, write_csv
from brainsentinel.data.palette import Palette, train_palette

SIZE = 240
LESION_RGB = (255, 255, 255)


def test_palette_colors() -> np.ndarray:
    i = np.arange(256)
    return np.stack([i, (7 * i) // 8, np.minimum(255, (7 * i) // 8 + i // 8)], axis=1).astype(np.uint8)


def _render(idx: np.ndarray, colors: np.ndarray) -> np.ndarray:
    rgb = colors[idx]
    return np.concatenate([rgb, np.full(idx.shape + (1,), 255, np.uint8)], axis=2)


def make_slice(rng, lesion: bool):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cy, cx = 120 + rng.integers(-6, 7), 120 + rng.integers(-6, 7)
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 < 88 ** 2
    a, b = rng.normal(0, 0.4, 2)
    tex = 40 * np.sin((xx + 25 * a) / 18.0) + 30 * np.cos((yy + 25 * b) / 23.0) + rng.normal(0, 4, (SIZE, SIZE))
    idx = np.where(disc, np.clip(110 + tex, 1, 255), 0).astype(np.int16)
    mask = np.zeros((SIZE, SIZE), bool)
    if lesion:
        ly, lx = cy + rng.integers(-35, 36), cx + rng.integers(-35, 36)
        r = rng.integers(7, 16)
        mask = ((yy - ly) ** 2 + (xx - lx) ** 2 < r ** 2) & disc
        idx = np.where(mask, np.clip(idx + 100, 1, 255), idx)
    return idx.astype(np.uint8), mask


def full_range_slice(rng):
    """One train image whose disc uses every intensity 1..255 so the train palette has all 256 colours
    (the production palette guard demands that; it is not loosened for tests)."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    disc = (yy - 120) ** 2 + (xx - 120) ** 2 < 88 ** 2
    idx = np.where(disc, rng.integers(1, 256, (SIZE, SIZE)), 0).astype(np.uint8)
    return idx, np.zeros((SIZE, SIZE), bool)


def build(root: Path, seed: int = 0, n_train_patients: int = 12, n_valid: int = 3, n_test: int = 4):
    """Create root/Brain/... and return (cfg, palette). Patients are disjoint across splits (no overlap)."""
    rng = np.random.default_rng(seed)
    colors = test_palette_colors()
    layout = {  # key -> (patients, slices, lesion, has_mask_dir)
        "train/good": (range(100, 100 + n_train_patients), (60, 68), False),
        "valid/good": (range(200, 200 + n_valid), (60, 92), False),
        "valid/Ungood": (range(300, 300 + n_valid), (60, 68), True),
        "test/good": (range(400, 400 + n_test), (60, 92), False),
        "test/Ungood": (range(500, 500 + n_test), (60, 68), True),
    }
    splits, brain = {}, root / "Brain"
    for key, (pats, sls, les) in layout.items():
        flat = key == "train/good"
        img_dir = brain / key if flat else brain / key / "img"
        img_dir.mkdir(parents=True, exist_ok=True)
        if not flat:
            (brain / key / "label").mkdir(parents=True, exist_ok=True)
        n = 0
        if flat:
            idx, _ = full_range_slice(rng)
            Image.fromarray(_render(idx, colors), "RGBA").save(img_dir / "00099_60.png")
            n += 1
        for p in pats:
            for s in sls:
                idx, mask = make_slice(rng, les)
                Image.fromarray(_render(idx, colors), "RGBA").save(img_dir / f"{p:05d}_{s}.png")
                if not flat:
                    m = np.zeros((SIZE, SIZE, 4), np.uint8)
                    m[..., 3] = 255
                    m[mask, :3] = LESION_RGB
                    Image.fromarray(m, "RGBA").save(brain / key / "label" / f"{p:05d}_{s}.png")
                n += 1
        splits[key] = {"img_dir": "." if flat else "img", "mask_dir": None if flat else "label", "expected_n": n}
    cfg = {"root": str(brain), "image_size": SIZE, "seed": 0, "splits": splits,
           "mask": {"allowed_rgb": [[0, 0, 0], [56, 55, 77], [112, 123, 143], [255, 255, 255]], "interpolation": "nearest"},
           "patient_overlap": {"valid_test_ids": []}, "known_issues": {"valid_test_duplicate_files": [],
                                                                       "sensitivity_exclusion_expected_test_images": 0}}
    pal = train_palette(cfg)
    rows = build_rows(cfg, pal)
    add_flags(rows)
    write_csv(rows, columns(cfg), root / "index.csv")
    return cfg, pal
