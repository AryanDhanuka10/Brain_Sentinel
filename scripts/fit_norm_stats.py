"""Fit scalar mean/std on train/good BRAIN pixels only and save JSON. Run from repo root:
    python scripts/fit_norm_stats.py
Value fitted = (idx - corner_bg)/255 over idx > corner_bg. Valid/test are never read.
The default mode (imagenet constants) does not need this file; it is for the autoencoder / ablations.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from brainsentinel.data.checks import list_images, load_config  # noqa: E402
from brainsentinel.data.palette import train_palette  # noqa: E402
from brainsentinel.data.preprocessing import (  # noqa: E402
    file_list_sha256, load_index_image, load_pp_config, resolve_cfg)


def main() -> None:
    cfg = resolve_cfg(load_config(REPO / "configs" / "data.yaml"))
    pp = load_pp_config()
    pal = train_palette(cfg)
    files = list_images(cfg, "train/good")
    s = ss = 0.0
    n = 0
    for p in files:
        idx, brain, _ = load_index_image(p, pal, pp["input"]["size"])
        v = idx[brain].astype(np.float64) / 255.0
        s += float(v.sum())
        ss += float((v * v).sum())
        n += int(v.size)
    mean = s / n
    std = max(ss / n - mean * mean, 0.0) ** 0.5
    out = {"fitted_on": "train/good", "n_images": len(files), "n_pixels": n,
           "pixel_set": "brain (idx > corner_bg), offset removed, value = idx/255",
           "file_list_sha256": file_list_sha256(files), "mean": mean, "std": std}
    path = REPO / pp["normalize"]["train_stats_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
