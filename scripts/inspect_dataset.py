"""Day 2 step 3: contour overlays, lesion-on-black, intensity stats by group and slice bin."""
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
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from brainsentinel.data.checks import load_config, mask_path, mask_to_binary, split_dir  # noqa: E402
from brainsentinel.data.palette import Palette, brain_mask, corner_bg, load_rgb, pack_rgb  # noqa: E402

OUT = ROOT / "artifacts" / "day2"
GROUPS = ["train/good", "valid/good", "valid/Ungood", "test/good", "test/Ungood"]
BINS = ["60s", "70s", "80s", "90s"]


def emd(a: np.ndarray, b: np.ndarray) -> float:
    """1-D earth mover's distance between two count histograms, in index units (0..255)."""
    return float(np.abs(np.cumsum(a / a.sum()) - np.cumsum(b / b.sum())).sum())


def main() -> int:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((ROOT / "data/processed/index.csv").open()))
    for r in rows:
        r["g"] = f"{r['split']}/{r['class']}"
    path_of = lambda r: split_dir(cfg, r["g"], "img") / r["file"]  # noqa: E731

    seen = np.zeros(1 << 24, dtype=bool)  # palette from TRAIN only
    for r in rows:
        if r["split"] == "train":
            seen[pack_rgb(load_rgb(path_of(r))).ravel()] = True
    pal = Palette.from_keys(np.flatnonzero(seen))
    assert tuple(pal.colors[0]) == (0, 0, 0), f"index 0 is {pal.colors[0]}, not black"

    tol = max(int(r["corner_bg"]) for r in rows if r["split"] == "train")  # from TRAIN only
    hist = defaultdict(lambda: np.zeros(256, dtype=np.int64))  # (group, bin) -> brain-pixel counts
    bf = defaultdict(list)  # (group, bin) -> brain fraction per image
    corner_bad = 0
    black = {"lesion": defaultdict(int), "on_black": defaultdict(int)}
    per_img_black: dict[tuple, float] = {}
    for r in rows:
        idx = pal.encode(load_rgb(path_of(r)))
        corner_bad += int(corner_bg(idx) > tol)
        brain = brain_mask(idx)
        k = (r["g"], r["slice_bin"])
        hist[k] += np.bincount(idx[brain], minlength=256)
        bf[k].append(float(brain.mean()))
        if r["class"] == "Ungood":
            m = mask_to_binary(np.asarray(Image.open(mask_path(cfg, r["g"], path_of(r))).convert("RGBA")))
            n, nb = int(m.sum()), int(((m == 1) & ~brain).sum())
            black["lesion"][r["g"]] += n
            black["on_black"][r["g"]] += nb
            per_img_black[(r["g"], r["file"])] = nb / n

    res: dict = {"palette_index0_black": True, "images_with_nonblack_corner": corner_bad}
    res["lesion_px_on_index0"] = {g: {"on_black": black["on_black"][g], "lesion": black["lesion"][g],
                                      "share": black["on_black"][g] / black["lesion"][g],
                                      "images_with_any": sum(1 for (gg, _), v in per_img_black.items() if gg == g and v > 0),
                                      "max_per_image_share": max(v for (gg, _), v in per_img_black.items() if gg == g)}
                                  for g in ("valid/Ungood", "test/Ungood")}
    allh = {g: sum((hist[(g, b)] for b in BINS), np.zeros(256, dtype=np.int64)) for g in GROUPS}
    res["brain_fraction_mean_n"] = {g: {b: [round(float(np.mean(bf[(g, b)])), 4), len(bf[(g, b)])]
                                        for b in BINS if bf[(g, b)]} for g in GROUPS}
    res["emd_vs_train_good_overall"] = {g: round(emd(allh["train/good"], allh[g]), 3) for g in GROUPS[1:]}
    res["emd_vs_train_good_same_bin"] = {g: {b: round(emd(hist[("train/good", b)], hist[(g, b)]), 3)
                                             for b in BINS if bf[(g, b)]} for g in GROUPS[1:]}
    pat = lambda g: {r["patient"] for r in rows if r["g"] == g}  # noqa: E731
    res["good_vs_ungood_patient_overlap"] = {s: len(pat(f"{s}/good") & pat(f"{s}/Ungood")) for s in ("valid", "test")}
    res["patients_per_group"] = {g: len(pat(g)) for g in GROUPS}
    (ROOT / "results" / "day2_inspect.json").write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for g in GROUPS:
        ax[0].plot(allh[g] / allh[g].sum(), label=g, lw=1)
    ax[0].set(title="brain-pixel intensity index (index>corner bg), normalized", xlabel="index", ylabel="density")
    ax[0].legend(fontsize=7)
    for g in GROUPS:
        ax[1].plot(BINS, [np.mean(bf[(g, b)]) if bf[(g, b)] else np.nan for b in BINS], marker="o", label=g)
    ax[1].set(title="mean brain fraction by slice bin", ylabel="fraction of frame")
    fig.tight_layout()
    fig.savefig(OUT / "intensity_stats.png", dpi=130)
    plt.close(fig)

    rng = np.random.default_rng(cfg["seed"])
    for g in ("valid/Ungood", "test/Ungood"):
        sub = sorted((r for r in rows if r["g"] == g), key=lambda r: int(r["lesion_px"]))
        pick = sub[:2] + sub[-2:]
        worst = sorted(sub, key=lambda r: -per_img_black[(g, r["file"])])[:2]
        pick += [r for r in worst if r not in pick]
        rest = [r for r in sub if r not in pick]
        pick += [rest[i] for i in rng.choice(len(rest), 4, replace=False)]
        cols = 5
        fig, axs = plt.subplots(-(-len(pick) // cols), cols, figsize=(3 * cols, 3.1 * -(-len(pick) // cols)))
        for a in np.atleast_1d(axs).ravel():
            a.axis("off")
        for a, r in zip(np.atleast_1d(axs).ravel(), pick):
            a.imshow(load_rgb(path_of(r)))
            m = mask_to_binary(np.asarray(Image.open(mask_path(cfg, g, path_of(r))).convert("RGBA")))
            a.contour(m, levels=[0.5], colors="red", linewidths=0.8)
            a.set_title(f"{r['file']} {r['lesion_px']}px\nblack={per_img_black[(g, r['file'])]:.2f}", fontsize=7)
        fig.suptitle(f"{g}: 2 smallest, 2 largest, worst-on-black, 4 random (seed {cfg['seed']})")
        fig.tight_layout()
        fig.savefig(OUT / f"overlay_{g.replace('/', '_')}.png", dpi=130)
        plt.close(fig)

    fig, axs = plt.subplots(3, 5, figsize=(15, 9.5))
    for row, g in zip(axs, ("train/good", "valid/good", "test/good")):
        sub = [r for r in rows if r["g"] == g]
        for a, i in zip(row, rng.choice(len(sub), 5, replace=False)):
            a.imshow(load_rgb(path_of(sub[i])))
            a.set_title(f"{g} {sub[i]['file']}", fontsize=7)
            a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "normals_montage.png", dpi=130)
    plt.close(fig)

    print(json.dumps(res, indent=2))
    ok = corner_bad == 0
    print("RESULT:", "PASS" if ok else f"FAIL corners non-black in {corner_bad} images")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
