#!/usr/bin/env python
"""Diagnose valid/Ungood <-> test/Ungood duplicates. Read-only."""
import hashlib
import json
from pathlib import Path

R = Path("data/raw/Brain")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
names = lambda k, sub: {p.name: p for p in (R / k / sub).glob("*.png")}  # noqa: E731
sl = lambda ns, pid: sorted(int(n.split("_")[1][:-4]) for n in ns if n.startswith(pid + "_"))  # noqa: E731

v, t, tg, vg = names("valid/Ungood", "img"), names("test/Ungood", "img"), names("test/good", "img"), names("valid/good", "img")
shared = sorted(set(v) & set(t))
rows = []
for n in shared:
    pid = n.split("_")[0]
    rows.append({
        "file": n,
        "image_identical": sha(v[n]) == sha(t[n]),
        "mask_identical": sha(R / "valid/Ungood/label" / n) == sha(R / "test/Ungood/label" / n),
        "valid_Ungood_slices": sl(v, pid), "test_Ungood_slices": sl(t, pid),
        "valid_good_slices": sl(vg, pid), "test_good_slices": sl(tg, pid),
    })
out = {
    "n_shared_filenames": len(shared),
    "n_image_identical": sum(r["image_identical"] for r in rows),
    "n_mask_identical": sum(r["mask_identical"] for r in rows),
    "frac_of_valid_Ungood": len(shared) / len(v),
    "frac_of_test_Ungood": len(shared) / len(t),
    "rows": rows,
}
Path("results/day1_cross_split_duplicates.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
