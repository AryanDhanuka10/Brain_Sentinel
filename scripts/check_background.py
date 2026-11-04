"""Day 2 step 5 (descriptive): background-offset images and border uniformity."""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from brainsentinel.data.checks import load_config, split_dir  # noqa: E402
from brainsentinel.data.palette import brain_mask, corner_bg, load_rgb, train_palette  # noqa: E402


def main() -> int:
    cfg = load_config()
    pal = train_palette(cfg)
    rows = list(csv.DictReader(open(ROOT / "data/processed/index.csv")))
    n = cfg["image_size"]
    frame = np.zeros((n, n), dtype=bool)
    frame[:8] = frame[-8:] = True
    frame[:, :8] = frame[:, -8:] = True

    share = defaultdict(list)
    off, tot = [], Counter()
    for r in rows:
        g = f"{r['split']}/{r['class']}"
        path = split_dir(cfg, g, "img") / r["file"]
        idx = pal.encode(load_rgb(path))
        bg = corner_bg(idx)
        s = float((idx[frame] == bg).mean())
        share[g].append(s)
        tot[r["patient"]] += 1
        if bg > 0:
            off.append({"group": g, "file": r["file"], "patient": r["patient"], "bg": bg,
                        "border_eq_bg": round(s, 4), "frac_old": round(float((idx > 0).mean()), 3),
                        "frac_new": round(float(brain_mask(idx).mean()), 3), "path": str(path)})

    out = {"border_uniformity": {g: {"n": len(v), "min": round(min(v), 4),
                                     "n_below_0.99": int(sum(x < 0.99 for x in v))} for g, v in share.items()},
           "offset_patients_flagged_of_total": {p: [sum(1 for o in off if o["patient"] == p), tot[p]]
                                                for p in sorted({o["patient"] for o in off})},
           "offset_images": [{k: v for k, v in o.items() if k != "path"} for o in off]}
    (ROOT / "results" / "day2_background.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "offset_images"}, indent=2))
    for o in out["offset_images"]:
        print(o)

    pick = [o for o in off if o["patient"] in ("01551", "00446")][:8]
    if pick:
        fig, axs = plt.subplots(2, 4, figsize=(14, 7.4))
        for a in axs.ravel():
            a.axis("off")
        for a, o in zip(axs.ravel(), pick):
            idx = pal.encode(load_rgb(Path(o["path"])))
            a.imshow(load_rgb(Path(o["path"])))
            a.contour(brain_mask(idx), levels=[0.5], colors="yellow", linewidths=0.6)
            a.set_title(f"{o['group']} {o['file']} bg={o['bg']}\nfrac old {o['frac_old']} new {o['frac_new']}", fontsize=7)
        fig.tight_layout()
        fig.savefig(ROOT / "artifacts" / "day2" / "offset_other_patients.png", dpi=120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
