#!/usr/bin/env python
"""Day 1 I/O benchmark on this filesystem (fuseblk). Decides caching strategy on Day 3."""
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

files = sorted(Path("data/raw/Brain/train/good").glob("*.png"))
sample = random.Random(0).sample(files, 500)


def read_all():
    t = time.perf_counter()
    for f in sample:
        np.asarray(Image.open(f).convert("RGB"))
    return time.perf_counter() - t


t1, t2 = read_all(), read_all()  # pass 1 may be warm if the audit just read these files
mem = {}
for line in open("/proc/meminfo"):
    k, v = line.split(":")
    if k in ("MemTotal", "MemAvailable"):
        mem[k + "_GiB"] = round(int(v.split()[0]) / 2**20, 2)
res = {
    "n_files": len(sample),
    "pass1_ms_per_image": round(t1 / 500 * 1000, 2),
    "pass2_ms_per_image": round(t2 / 500 * 1000, 2),
    "est_full_train_epoch_read_s_pass1": round(t1 / 500 * 7500, 1),
    "train_cache_RGB_uint8_GiB": round(7500 * 240 * 240 * 3 / 2**30, 2),
    **mem,
}
Path("results").mkdir(exist_ok=True)
Path("results/day1_io_bench.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
