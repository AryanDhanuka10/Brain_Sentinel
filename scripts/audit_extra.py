#!/usr/bin/env python
"""Day 1 extra audit: colour structure, overlap by group, lesion area, slice index. Read-only."""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

R = Path("data/raw/Brain")
G = ["train/good", "valid/good", "valid/Ungood", "test/good", "test/Ungood"]
NAME = re.compile(r"^(\d+)_(\d+)\.png$")


def imgdir(k: str) -> Path:
    p = R / k / "img"
    return p if p.is_dir() else R / k


def main() -> None:
    rng, out = random.Random(0), {}

    triples, black, uniq = Counter(), [], []
    for k in G:
        fs = sorted(imgdir(k).glob("*.png"))
        for f in rng.sample(fs, min(100, len(fs))):
            a = np.asarray(Image.open(f))[..., :3].reshape(-1, 3)
            black.append(float((a.max(1) == 0).mean()))
            u, c = np.unique(a, axis=0, return_counts=True)
            uniq.append(len(u))
            for t, n in zip(map(tuple, u.tolist()), c):
                triples[t] += int(n)
    tot = sum(triples.values())
    out["image_colour"] = {
        "n_images_sampled": len(uniq),
        "n_unique_rgb_triples_overall": len(triples),
        "frac_pixels_in_top256_triples": sum(c for _, c in triples.most_common(256)) / tot,
        "median_unique_triples_per_image": float(np.median(uniq)),
        "mean_frac_pure_black_pixels": float(np.mean(black)),
        "most_common_10": [[list(t), c] for t, c in triples.most_common(10)],
    }

    ids = {k: {m.group(1) for f in imgdir(k).glob("*.png") if (m := NAME.match(f.name))} for k in G}
    pairs = [("valid/good", "test/good"), ("valid/good", "test/Ungood"),
             ("valid/Ungood", "test/good"), ("valid/Ungood", "test/Ungood")]
    out["valid_test_overlap_by_group"] = {f"{a}&{b}": sorted(ids[a] & ids[b]) for a, b in pairs}

    for k in ["valid/Ungood", "test/Ungood"]:
        ar = np.array([int((np.asarray(Image.open(f))[..., :3].max(-1) > 0).sum())
                       for f in sorted((R / k / "label").glob("*.png"))])
        out[f"mask_area_px__{k}"] = {
            "n": len(ar), "pct_5_25_50_75_95": np.percentile(ar, [5, 25, 50, 75, 95]).tolist(),
            "min": int(ar.min()), "max": int(ar.max()),
            "lt_10px": int((ar < 10).sum()), "lt_50px": int((ar < 50).sum()), "lt_100px": int((ar < 100).sum()),
        }

    out["slice_index"] = {}
    for k in G:
        s = [int(NAME.match(f.name).group(2)) for f in imgdir(k).glob("*.png")]
        out["slice_index"][k] = {"min": min(s), "max": max(s), "mean": float(np.mean(s)),
                                 "hist_by_10": dict(sorted(Counter((x // 10) * 10 for x in s).items()))}

    Path("results").mkdir(exist_ok=True)
    Path("results/day1_audit_extra.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
