#!/usr/bin/env python
"""Day 1 audit of the BMAD Brain subset (RGBA-aware). Read-only."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

NAME_RE = re.compile(r"^(\d+)_(\d+)\.png$")
MASK_DIR_NAMES = ("label", "anomaly_mask")
GROUPS = ["train/good", "valid/good", "valid/Ungood", "test/good", "test/Ungood"]


def list_pngs(d: Path) -> list[Path]:
    return sorted(d.glob("*.png")) if d.is_dir() else []


def header_scan(files: list[Path]) -> dict:
    c: Counter = Counter()
    for f in files:
        with Image.open(f) as im:
            c[f"{im.size[0]}x{im.size[1]}|{im.mode}"] += 1
    return dict(c)


def image_stats(files: list[Path], n: int, rng: random.Random) -> dict:
    sample = rng.sample(files, min(n, len(files)))
    gray_like, dtypes, alpha_vals = 0, set(), set()
    mins, maxs, nz = [], [], []
    for f in sample:
        a = np.asarray(Image.open(f))
        dtypes.add(str(a.dtype))
        if a.ndim == 3:
            rgb = a[..., :3]
            gray_like += int(np.array_equal(rgb[..., 0], rgb[..., 1])
                             and np.array_equal(rgb[..., 1], rgb[..., 2]))
            if a.shape[-1] == 4:
                alpha_vals.update(np.unique(a[..., 3]).tolist()[:5])
            v = rgb.max(axis=-1)
        else:
            gray_like += 1
            v = a
        mins.append(int(v.min())); maxs.append(int(v.max())); nz.append(float((v > 0).mean()))
    return {
        "n_sampled": len(sample),
        "dtypes": sorted(dtypes),
        "n_rgb_channels_identical": gray_like,   # == n_sampled -> effectively grayscale
        "alpha_values_seen": sorted(alpha_vals),
        "intensity_min": min(mins), "intensity_max": max(maxs),
        "mean_nonzero_frac": float(np.mean(nz)),
    }


def mask_stats(files: list[Path]) -> dict:
    colors: set[tuple] = set()
    fracs, empty, max_colors, alpha_vals = [], 0, 0, set()
    for f in files:
        a = np.asarray(Image.open(f))
        if a.ndim == 3:
            rgb = a[..., :3]
            if a.shape[-1] == 4:
                alpha_vals.update(np.unique(a[..., 3]).tolist()[:5])
            fg = rgb.max(axis=-1) > 0
            u = np.unique(rgb.reshape(-1, 3), axis=0)
        else:
            fg = a > 0
            u = np.unique(a)[:, None]
        colors.update(map(tuple, u.tolist()))
        max_colors = max(max_colors, len(u))
        fracs.append(float(fg.mean()))
        empty += int(not fg.any())
    return {
        "foreground_rule": "any RGB channel > 0 (alpha ignored) - TENTATIVE",
        "alpha_values_seen": sorted(alpha_vals),
        "n_distinct_rgb_union": len(colors),
        "max_distinct_rgb_in_one_mask": max_colors,
        "rgb_values_union_first12": sorted(colors)[:12],
        "n_empty_masks": empty,
        "fg_frac_min": float(np.min(fracs)),
        "fg_frac_median": float(np.median(fracs)),
        "fg_frac_max": float(np.max(fracs)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/Brain")
    ap.add_argument("--out", default="results/day1_dataset_audit.json")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root, rng = Path(args.root), random.Random(args.seed)
    report: dict = {"root": str(root), "groups": {}, "warnings": []}
    patients: dict[str, set[str]] = {"train": set(), "valid": set(), "test": set()}

    for key in GROUPS:
        g: dict = {}
        report["groups"][key] = g
        img_dir = root / key / "img"
        if not img_dir.is_dir():
            img_dir = root / key  # train/good is flat in this copy
        imgs = list_pngs(img_dir)
        g["img_dir"] = str(img_dir.relative_to(root))
        g["n_images"] = len(imgs)
        if not imgs:
            report["warnings"].append(f"{key}: no images at {img_dir}")
            continue

        split = key.split("/")[0]
        ids = [m.group(1) for f in imgs if (m := NAME_RE.match(f.name))]
        if len(ids) != len(imgs):
            report["warnings"].append(f"{key}: unexpected filenames")
        patients[split].update(ids)
        g["n_patients_parsed"] = len(set(ids))
        g["size_mode_counts"] = header_scan(imgs)
        g["image_stats"] = image_stats(imgs, args.sample, rng)

        mask_dir = next((root / key / n for n in MASK_DIR_NAMES if (root / key / n).is_dir()), None)
        if mask_dir is None:
            g["mask_dir"] = None
            continue
        masks = list_pngs(mask_dir)
        g["mask_dir"] = mask_dir.name
        g["n_masks"] = len(masks)
        a, b = {f.name for f in imgs}, {f.name for f in masks}
        g["images_without_mask"], g["masks_without_image"] = len(a - b), len(b - a)
        if a != b:
            report["warnings"].append(f"{key}: image/mask pairing mismatch")
        g["mask_size_mode_counts"] = header_scan(masks)
        g["mask_stats"] = mask_stats(masks)
        n_empty = g["mask_stats"]["n_empty_masks"]
        if key.endswith("/good") and n_empty != len(masks):
            report["warnings"].append(f"{key}: {len(masks) - n_empty} good masks have foreground")
        if key.endswith("/Ungood") and n_empty:
            report["warnings"].append(f"{key}: {n_empty} Ungood masks have NO foreground")

    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    report["patient_overlap_counts"] = {f"{x}&{y}": len(patients[x] & patients[y]) for x, y in pairs}
    report["patient_overlap_ids"] = {f"{x}&{y}": sorted(patients[x] & patients[y]) for x, y in pairs}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()