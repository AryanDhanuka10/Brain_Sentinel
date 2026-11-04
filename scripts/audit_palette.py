#!/usr/bin/env python
"""Day 1: is the image colour a lossless 256-entry lookup table? Reads every image. Read-only."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

R = Path("data/raw/Brain")
G = ["train/good", "valid/good", "valid/Ungood", "test/good", "test/Ungood"]


def imgdir(k: str) -> Path:
    p = R / k / "img"
    return p if p.is_dir() else R / k


def pack(a: np.ndarray) -> np.ndarray:
    return (a[..., 0].astype(np.uint32) << 16) | (a[..., 1].astype(np.uint32) << 8) | a[..., 2].astype(np.uint32)


per_group, all_vals, n_img, alpha_bad = {}, set(), 0, 0
for k in G:
    vals = set()
    for f in sorted(imgdir(k).glob("*.png")):
        a = np.asarray(Image.open(f))
        n_img += 1
        alpha_bad += int((a[..., 3] != 255).any())
        vals.update(np.unique(pack(a[..., :3])).tolist())
    per_group[k] = vals
    all_vals |= vals

trip = np.array([[(v >> 16) & 255, (v >> 8) & 255, v & 255] for v in sorted(all_vals)], dtype=np.int32)
lum = trip @ np.array([0.299, 0.587, 0.114])
t = trip[np.argsort(lum, kind="stable")]
out = {
    "n_images_read": n_img,
    "images_with_alpha_not_255": alpha_bad,
    "n_unique_rgb_triples_all_images": len(all_vals),
    "unique_per_group": {k: len(v) for k, v in per_group.items()},
    "all_groups_use_identical_set": all(v == all_vals for v in per_group.values()),
    "black_present": bool((trip.sum(1) == 0).any()),
    "n_gray_triples_R_eq_G_eq_B": int(((trip[:, 0] == trip[:, 1]) & (trip[:, 1] == trip[:, 2])).sum()),
    "monotone_non_decreasing_by_luminance": {c: bool((np.diff(t[:, i]) >= 0).all()) for i, c in enumerate("RGB")},
}
try:
    import matplotlib
    lut = np.rint(matplotlib.colormaps["bone"](np.arange(256))[:, :3] * 255).astype(np.int32)
    d = np.abs(trip[:, None, :] - lut[None, :, :]).max(-1)
    out["matches_matplotlib_bone"] = {
        "max_abs_diff_observed_to_nearest_lut": int(d.min(1).max()),
        "n_observed_diff_gt1": int((d.min(1) > 1).sum()),
        "every_lut_entry_matched_within_1": bool((d.min(0) <= 1).all()),
    }
except Exception as e:  # noqa: BLE001
    out["matches_matplotlib_bone"] = {"error": repr(e)[:100]}

Path("results").mkdir(exist_ok=True)
Path("results/day1_palette_check.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
