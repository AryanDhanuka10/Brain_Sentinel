#!/usr/bin/env python
"""Day 1 integrity audit: corrupt files, duplicate pixels, cross-split collisions. Read-only."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

R = Path("data/raw/Brain")
G = ["train/good", "valid/good", "valid/Ungood", "test/good", "test/Ungood"]
split = lambda k: k.split("/")[0]  # noqa: E731


def imgdir(k: str) -> Path:
    p = R / k / "img"
    return p if p.is_dir() else R / k


def main() -> None:
    by_hash, names, bad = defaultdict(list), defaultdict(set), []
    for k in G:
        for f in sorted(imgdir(k).glob("*.png")):
            if f.stat().st_size == 0:
                bad.append([k, f.name, "zero bytes"]); continue
            try:
                with Image.open(f) as im:
                    im.verify()
            except Exception as e:  # noqa: BLE001
                bad.append([k, f.name, repr(e)[:80]])
            by_hash[hashlib.sha256(f.read_bytes()).hexdigest()].append([k, f.name])
            names[f.name].add(k)

    within, cross = [], []
    for locs in by_hash.values():
        if len(locs) > 1:
            (cross if len({split(k) for k, _ in locs}) > 1 else within).append(locs)
    name_cross = {n: sorted(ks) for n, ks in names.items() if len({split(k) for k in ks}) > 1}

    out = {
        "n_files_hashed": sum(len(v) for v in by_hash.values()),
        "corrupt_or_zero_byte": {"count": len(bad), "examples": bad[:10]},
        "identical_pixels_within_split": {"groups": len(within), "examples": within[:10]},
        "identical_pixels_across_splits": {"groups": len(cross), "examples": cross[:10]},
        "same_filename_across_splits": {"count": len(name_cross), "examples": dict(list(name_cross.items())[:10])},
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/day1_integrity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

