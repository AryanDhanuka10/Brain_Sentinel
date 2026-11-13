"""Prediction collection, metric evaluation, latency/memory benchmarking and the test-split guard.

Protocol (docs/EVALUATION_METHODOLOGY.md): thresholds are selected on ``valid`` and frozen to
``threshold.json``; ``test`` is only evaluated against that frozen file and only once per run unless
``allow_rerun`` is passed (every evaluation is appended to ``evaluations.jsonl``).
"""
from __future__ import annotations

import csv
import json
import resource
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from .metrics import (PixelStats, dice_iou_at, image_auroc_aupr, image_threshold_metrics, patient_bootstrap_ci)
from .thresholding import freeze, load_frozen, select_image_threshold, select_pixel_threshold

META_COLS = ["file", "key", "patient", "slice", "slice_bin", "label", "score", "lesion_px", "corner_bg"]


# ------------------------------------------------------------------------------ predictions
@torch.no_grad()
def collect_predictions(model, batches: Iterable[dict], n: int, out_dir: str | Path, device: str = "cpu",
                        size: int = 240) -> Path:
    """Run ``model.predict`` over batches and store maps/masks/brain as .npy memmaps + meta.csv.

    A batch is a dict with ``image`` (B,3,H,W), optional ``mask`` (B,H,W), ``brain`` (B,H,W), and
    lists/tensors ``file, key, patient, slice, slice_bin, corner_bg``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    maps = np.lib.format.open_memmap(out / "maps.npy", "w+", np.float32, (n, size, size))
    masks = np.lib.format.open_memmap(out / "masks.npy", "w+", np.uint8, (n, size, size))
    brain = np.lib.format.open_memmap(out / "brain.npy", "w+", np.bool_, (n, size, size))
    rows, i = [], 0
    model.eval().to(device)
    for b in batches:
        m, s = model.predict(b["image"].to(device))
        k = m.shape[0]
        maps[i:i + k] = m.cpu().numpy()
        mk = b.get("mask")
        masks[i:i + k] = mk.numpy() if mk is not None else 0
        brain[i:i + k] = b["brain"].numpy() if "brain" in b else True
        lp = (mk.flatten(1).sum(1).tolist() if mk is not None else [None] * k)
        for j in range(k):
            key = b["key"][j] if not isinstance(b["key"], str) else b["key"]
            rows.append({"file": b["file"][j], "key": key, "patient": b["patient"][j],
                         "slice": int(b["slice"][j]), "slice_bin": b["slice_bin"][j],
                         "label": int(key.endswith("Ungood")), "score": float(s[j]),
                         "lesion_px": lp[j], "corner_bg": int(b["corner_bg"][j])})
        i += k
    if i != n:
        raise ValueError(f"expected {n} images, got {i}")
    for a in (maps, masks, brain):
        a.flush()
    with (out / "meta.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, META_COLS)
        w.writeheader()
        w.writerows(rows)
    return out


class IndexedView:
    """Read-only chunk view of a memmap restricted to ``idx`` (avoids copying 800 MB)."""

    def __init__(self, arr, idx):
        self.arr, self.idx = arr, np.asarray(idx)
        self.shape = (len(self.idx),) + tuple(arr.shape[1:])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        sel = self.idx[k]
        return self.arr[sel] if np.ndim(sel) else self.arr[int(sel)]


def load_predictions(pred_dir: str | Path) -> dict:
    d = Path(pred_dir)
    with (d / "meta.csv").open() as f:
        meta = list(csv.DictReader(f))
    return {"maps": np.load(d / "maps.npy", mmap_mode="r"), "masks": np.load(d / "masks.npy", mmap_mode="r"),
            "brain": np.load(d / "brain.npy", mmap_mode="r"), "meta": meta,
            "scores": np.array([float(r["score"]) for r in meta]),
            "labels": np.array([int(r["label"]) for r in meta]),
            "patients": np.array([r["patient"] for r in meta])}


# ------------------------------------------------------------------------------ evaluation
def _pixel_block(maps, masks, roi, pix_thr, n_bins) -> dict:
    st = PixelStats.build(maps, masks, roi=roi, n_bins=n_bins)
    out = {"pixel_auroc": st.pixel_auroc(), "pixel_aupr": st.pixel_aupr(), "aupro_fpr0.3": st.aupro(0.3),
           "n_lesion_px": st.n_pos, "n_regions": int(len(st.region_size))}
    if pix_thr is not None:
        out.update({f"{k}": v for k, v in dice_iou_at(maps, masks, pix_thr, roi=roi).items()})
    return out


def evaluate_predictions(pred_dir: str | Path, thresholds: dict | None = None, n_bins: int = 4096,
                         n_boot: int = 1000, exclude_patients: Sequence[str] = (),
                         small_lesion_px: int = 100) -> dict:
    """All metric variants for one split. ``thresholds`` is a frozen validation record or None."""
    P = load_predictions(pred_dir)
    s, y, pat = P["scores"], P["labels"], P["patients"]
    img_thr = thresholds["image_threshold"] if thresholds else None
    pix_thr = thresholds["pixel_threshold"] if thresholds else None
    lesion = np.array([float(r["lesion_px"] or 0) for r in P["meta"]])
    res: dict = {"n_images": int(len(y)), "n_abnormal": int(y.sum()), "n_normal": int((y == 0).sum())}

    def image_block(idx):
        ss, yy, pp = s[idx], y[idx], pat[idx]
        out = image_auroc_aupr(ss, yy)
        lo, hi = patient_bootstrap_ci(ss, yy, pp, lambda a, b: image_auroc_aupr(a, b)["image_auroc"], n_boot=n_boot)
        out["image_auroc_ci95_patient_bootstrap"] = [lo, hi]
        if img_thr is not None:
            out.update({f"img_{k}": v for k, v in image_threshold_metrics(ss, yy, img_thr).items()})
        return out

    all_idx = np.arange(len(y))
    res["official"] = {"image": image_block(all_idx),
                       "pixel_full_frame": _pixel_block(P["maps"], P["masks"], None, pix_thr, n_bins),
                       "pixel_brain_roi": _pixel_block(P["maps"], P["masks"], P["brain"], pix_thr, n_bins)}
    # abnormal images with tiny lesions removed (normals kept): is localisation carried by large lesions?
    big = np.flatnonzero((y == 0) | (lesion >= small_lesion_px))
    if len(big) < len(y):
        res["without_lesions_under_%dpx" % small_lesion_px] = {
            "n_images": int(len(big)),
            "pixel_full_frame": _pixel_block(IndexedView(P["maps"], big), IndexedView(P["masks"], big), None, pix_thr, n_bins)}
    # slice-stratified image AUROC (position is confounded with the label; see data/README.md)
    strat = {}
    bins = np.array([r["slice_bin"] for r in P["meta"]])
    for b in sorted(set(bins)):
        ix = np.flatnonzero(bins == b)
        n1, n0 = int(y[ix].sum()), int((y[ix] == 0).sum())
        entry = {"n_normal": n0, "n_abnormal": n1}
        if n0 >= 5 and n1 >= 5:
            entry.update(image_auroc_aupr(s[ix], y[ix]))
        else:
            entry["note"] = "not computed: fewer than 5 images in one class"
        strat[b] = entry
    res["slice_stratified_image"] = strat
    if exclude_patients:
        keep = np.flatnonzero(~np.isin(pat, list(exclude_patients)))
        res["patient_disjoint"] = {"n_images": int(len(keep)), "n_excluded": int(len(y) - len(keep)),
                                   "image": image_block(keep),
                                   "pixel_full_frame": _pixel_block(IndexedView(P["maps"], keep),
                                                                    IndexedView(P["masks"], keep), None, pix_thr, n_bins)}
    return res


def select_and_freeze(valid_pred_dir: str | Path, out_json: str | Path, n_bins: int = 4096,
                      image_rule: str = "max_f1", pixel_rule: str = "max_dice") -> dict:
    """Choose both thresholds on the validation predictions and freeze them (valid only)."""
    P = load_predictions(valid_pred_dir)
    it = select_image_threshold(P["scores"], P["labels"], image_rule)
    pt = select_pixel_threshold(PixelStats.build(P["maps"], P["masks"], n_bins=n_bins), pixel_rule)
    return freeze(out_json, split="valid", image_threshold=it, image_rule=image_rule, pixel_threshold=pt,
                  pixel_rule=pixel_rule, n_images=len(P["labels"]))


# ------------------------------------------------------------------------------ test guard
def guard_test_run(run_dir: str | Path, allow_rerun: bool = False) -> dict:
    """Return the frozen validation thresholds or raise. Enforces: thresholds exist, test run once."""
    d = Path(run_dir)
    if not (d / "threshold.json").exists():
        raise PermissionError(f"{d}: no threshold.json. Evaluate on 'valid' first so thresholds are frozen there.")
    thr = load_frozen(d / "threshold.json")
    log = d / "evaluations.jsonl"
    if log.exists() and not allow_rerun and any(json.loads(l).get("split") == "test" for l in log.read_text().splitlines() if l):
        raise PermissionError(f"{d}: test split already evaluated once; pass allow_rerun to repeat (it will be logged).")
    return thr


def log_evaluation(run_dir: str | Path, split: str, extra: dict | None = None) -> None:
    with (Path(run_dir) / "evaluations.jsonl").open("a") as f:
        f.write(json.dumps({"split": split, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {})}) + "\n")


# ------------------------------------------------------------------------------ engineering metrics
def benchmark_latency(model, sample: torch.Tensor, device: str = "cpu", warmup: int = 5, runs: int = 50) -> dict:
    """Per-slice (batch=1) latency of ``model.predict`` on a device-resident tensor.

    Includes the backbone/network forward, distance computation, upsampling and smoothing. Excludes disk
    I/O, PNG decoding, palette recovery and host-to-device copies. Peak VRAM covers this process since
    the last reset.
    """
    x = sample[:1].to(device)
    model.eval().to(device)
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        model.predict(x)
    sync()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(x)
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"latency_ms_mean": float(np.mean(ts)), "latency_ms_p95": float(np.percentile(ts, 95)),
            "throughput_img_per_s": float(1000 / np.mean(ts)), "runs": runs, "device": device,
            "peak_vram_gb": (torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else None),
            "peak_cpu_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
