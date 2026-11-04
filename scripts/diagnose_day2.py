"""Day 2 diagnosis (descriptive only): non-black corners, low-index spike, image-level stats."""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from brainsentinel.data.checks import load_config, split_dir  # noqa: E402
from brainsentinel.data.palette import Palette, load_rgb, pack_rgb  # noqa: E402

GROUPS = ["train/good", "valid/good", "valid/Ungood", "test/good", "test/Ungood"]


def null_range(vals, pats, n_pat, per_pat, reps, rng):
    """2.5/50/97.5 percentile of the median of `vals` over random train patients x slices."""
    by = defaultdict(list)
    for v, p in zip(vals, pats):
        by[p].append(v)
    keys = [k for k, v in by.items() if len(v) >= per_pat]
    meds = []
    for _ in range(reps):
        ps = rng.choice(len(keys), n_pat, replace=False)
        meds.append(np.median([x for i in ps for x in rng.choice(by[keys[i]], per_pat, replace=False)]))
    return [round(float(x), 2) for x in np.percentile(meds, [2.5, 50, 97.5])]


def main() -> int:
    cfg = load_config()
    rows = list(csv.DictReader(open(ROOT / "data/processed/index.csv")))
    for r in rows:
        r["g"] = f"{r['split']}/{r['class']}"
    path_of = lambda r: split_dir(cfg, r["g"], "img") / r["file"]  # noqa: E731

    seen = np.zeros(1 << 24, dtype=bool)  # palette from TRAIN only
    for r in rows:
        if r["split"] == "train":
            seen[pack_rgb(load_rgb(path_of(r))).ravel()] = True
    pal = Palette.from_keys(np.flatnonzero(seen))

    F = defaultdict(lambda: defaultdict(list))
    corner_cases = []
    for r in rows:
        idx = pal.encode(load_rgb(path_of(r)))
        g = r["g"]
        c = [int(idx[0, 0]), int(idx[0, -1]), int(idx[-1, 0]), int(idx[-1, -1])]
        if any(c):
            corner_cases.append((g, r["file"], c))
        brain = idx > 0  # HYPOTHESIS: brain = index > 0
        nb = int(brain.sum())
        gy, gx = np.gradient(idx.astype(np.float32))
        F[g]["mean"].append(float(idx[brain].mean()) if nb else np.nan)
        F[g]["sharp"].append(float(np.hypot(gx, gy)[brain].mean()) if nb else np.nan)
        F[g]["low"].append(float(((idx >= 1) & (idx <= 8)).sum() / nb) if nb else np.nan)
        F[g]["pat"].append(r["patient"])

    q = lambda a: [round(float(x), 3) for x in np.nanpercentile(a, [25, 50, 75])]  # noqa: E731
    out: dict = {"n_corner_cases": len(corner_cases),
                 "corner_cases_by_group": {g: sum(1 for c in corner_cases if c[0] == g) for g in GROUPS},
                 "groups": {}}
    for g in GROUPS:
        d = F[g]
        out["groups"][g] = {"n": len(d["mean"]), "mean_brain_idx_q25_50_75": q(d["mean"]),
                            "grad_in_brain_q25_50_75": q(d["sharp"]),
                            "share_idx1to8_in_brain_q25_50_75": q(d["low"])}
    rng = np.random.default_rng(cfg["seed"])
    tr = F["train/good"]
    out["train_resampling_median_mean_brain_idx_2.5_50_97.5"] = {
        "39 patients x 1 slice (cf. valid/good)": null_range(tr["mean"], tr["pat"], 39, 1, 2000, rng),
        "11 patients x 4 slices (cf. valid/Ungood)": null_range(tr["mean"], tr["pat"], 11, 4, 2000, rng)}
    (ROOT / "results" / "day2_diagnose.json").write_text(json.dumps(out, indent=2))

    print("corner cases:", len(corner_cases))
    for g, f, c in corner_cases[:40]:
        print(" ", g, f, c)
    print(json.dumps(out, indent=2))
    pick = corner_cases[:8]
    if pick:
        (ROOT / "artifacts" / "day2").mkdir(parents=True, exist_ok=True)
        fig, axs = plt.subplots(2, 4, figsize=(14, 7.4))
        for a in axs.ravel():
            a.axis("off")
        for a, (g, f, c) in zip(axs.ravel(), pick):
            a.imshow(load_rgb(split_dir(cfg, g, "img") / f))
            a.set_title(f"{g} {f}\ncorners {c}", fontsize=7)
        fig.tight_layout()
        fig.savefig(ROOT / "artifacts" / "day2" / "corner_cases.png", dpi=120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
