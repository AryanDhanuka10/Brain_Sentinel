"""Train / evaluate / select orchestration used by scripts/*.py (and by the tests).

Everything that decides *which data a step may touch* lives here:

* ``train_run``   -> reads ``train/good`` only (+ ``valid/good`` NORMAL slices for autoencoder early stopping).
* ``evaluate_run``-> ``valid`` (selects + freezes thresholds) or ``test`` (needs frozen valid thresholds,
                     runs once unless ``allow_rerun``).
* ``select_model``-> reads only ``metrics_valid.json`` files.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .data.checks import list_images, load_config
from .data.dataset import BMADBrain, make_loader
from .data.palette import Palette, train_palette
from .data.preprocessing import REPO, file_list_sha256, get_norm, load_pp_config, resolve_cfg
from .evaluation import runner
from .evaluation.thresholding import load_frozen
from .models import build_model, load_model, save_model

VALID_KEYS = ["valid/good", "valid/Ungood"]
TEST_KEYS = ["test/good", "test/Ungood"]
SELECTION_MARGIN_AUPRO = 0.02            # predeclared: models within this AUPRO of the best are "tied"
MAX_VRAM_GB, MAX_LATENCY_MS = 6.0, 1000.0  # predeclared feasibility gates for the 8 GB laptop demo


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def device_str(pref: str = "auto") -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if pref == "auto" else pref


# ------------------------------------------------------------------------------ context
class Context:
    """Resolved data config, preprocessing config, train-derived palette, index path."""

    def __init__(self, data_cfg: dict, pp: dict, pal: Palette, index_csv: Path):
        self.cfg, self.pp, self.pal, self.index_csv = resolve_cfg(data_cfg), pp, pal, Path(index_csv)

    @classmethod
    def from_files(cls, data_config: str | Path | None = None, pp_config: str | Path | None = None,
                   palette_cache: str | Path | None = None) -> "Context":
        cfg = resolve_cfg(load_config(data_config or REPO / "configs" / "data.yaml"))
        pp = load_pp_config(pp_config)
        cache = Path(palette_cache or REPO / "artifacts" / "palette.json")
        pal = load_palette(cache) if cache.exists() else export_palette(cfg, cache)
        idx = Path(pp["index_csv"])
        return cls(cfg, pp, pal, idx if idx.is_absolute() else REPO / idx)

    def dataset(self, keys, exclude_patients=()):
        return BMADBrain(self.cfg, self.pp, self.pal, keys, index_csv=self.index_csv, exclude_patients=exclude_patients)


def export_palette(cfg: dict, path: str | Path) -> Palette:
    """Derive the 256-colour palette from train/good ONLY and cache it (also used by inference)."""
    pal = train_palette(cfg)
    files = list_images(cfg, "train/good")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"derived_from": "train/good", "n_images": len(files),
                             "file_list_sha256": file_list_sha256(files),
                             "colors": pal.colors.tolist()}) + "\n")
    return pal


def load_palette(path: str | Path) -> Palette:
    d = json.loads(Path(path).read_text())
    if d.get("derived_from") != "train/good":
        raise ValueError(f"{path}: palette not derived from train/good")
    colors = np.asarray(d["colors"], dtype=np.uint8)
    pal = Palette(colors)
    if len(colors) != 256 or not pal.is_monotone():
        raise ValueError(f"{path}: palette must be 256 monotone colours")
    return pal


# ------------------------------------------------------------------------------ logging tables
METRICS_COLS = ["run_id", "split", "model", "image_auroc", "image_aupr", "pixel_auroc", "aupro",
                "dice_val_thr", "iou_val_thr", "fpr", "latency_ms", "peak_vram_gb", "status"]
EXPERIMENT_COLS = ["run_id", "date", "model", "config_path", "seed", "checkpoint", "metrics_file",
                   "runtime_s", "peak_vram_gb", "status", "failure_notes"]


def append_row(path: str | Path, cols: list[str], row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists() or p.stat().st_size == 0
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, cols)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})


def run_id_for(kind: str, tag: str, seed: int) -> str:
    return f"{kind}_{tag}_s{seed}" if tag else f"{kind}_s{seed}"


# ------------------------------------------------------------------------------ training
def train_run(ctx: Context, kind: str, hparams: dict, fit_kwargs: dict, out_dir: str | Path, seed: int = 0,
              device: str = "auto", batch_size: int = 16, tag_config_path: str = "", log=print) -> dict:
    """Fit one model on the normal training split and write the run directory."""
    dev = device_str(device)
    seed_everything(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mean, std = get_norm(ctx.pp)
    train_ds = ctx.dataset(["train/good"])
    # shuffled (seeded) so any max_images subsample spans many patients, not the first few by filename
    loader = make_loader(train_ds, batch_size, shuffle=True, seed=seed)
    model = build_model(kind, hparams)
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    if kind == "autoencoder":
        val_loader = make_loader(ctx.dataset(["valid/good"]), batch_size)      # NORMAL validation slices only
        info = model.fit(loader, val_loader=val_loader, device=dev, log=log, **fit_kwargs)
    else:
        info = model.fit(loader, device=dev, log=log, **fit_kwargs)
    runtime = time.perf_counter() - t0
    size = save_model(model, out)
    if kind == "autoencoder":
        pd_hist = info.pop("history")
        with (out / "train_log.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, ["epoch", "train_mse", "val_normal_mse"])
            w.writeheader()
            w.writerows(pd_hist)
    peak = torch.cuda.max_memory_allocated() / 2**30 if dev.startswith("cuda") else None
    cfgj = {"run_id": out.name, "model": kind, "hparams": model.hparams(), "fit_kwargs": fit_kwargs, "seed": seed,
            "normalize": {"mode": ctx.pp["normalize"]["mode"], "mean": list(mean), "std": list(std)},
            "input_size": int(ctx.pp["input"]["size"]), "trained_on": "train/good",
            "n_train_available": len(train_ds), "fit_info": {k: v for k, v in info.items() if k != "grid"},
            "runtime_s": runtime, "peak_vram_gb": peak, "checkpoint_mb": size / 2**20, "device": dev,
            "torch": torch.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if dev.startswith("cuda") else None,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "config_path": tag_config_path}
    (out / "config.json").write_text(json.dumps(cfgj, indent=2, default=str) + "\n")
    # The run must carry the palette of the context it was trained with. Copy the cached file (it has provenance)
    # only when it is identical to ctx.pal; never pick up an unrelated artifacts/palette.json.
    cached = REPO / "artifacts" / "palette.json"
    same = False
    if cached.exists():
        try:
            same = bool(np.array_equal(load_palette(cached).colors, ctx.pal.colors))
        except Exception:  # noqa: BLE001 - unreadable cache: fall back to dumping ctx.pal
            same = False
    (out / "palette.json").write_text(cached.read_text() if same
                                      else json.dumps({"derived_from": "train/good", "colors": ctx.pal.colors.tolist()}))
    return cfgj


# ------------------------------------------------------------------------------ evaluation
def evaluate_run(ctx: Context, run_dir: str | Path, split: str, device: str = "auto", batch_size: int = 8,
                 n_bins: int = 4096, n_boot: int = 1000, allow_rerun: bool = False, bench: bool = True,
                 log=print) -> dict:
    if split not in ("valid", "test"):
        raise ValueError("split must be valid|test")
    d = Path(run_dir)
    dev = device_str(device)
    model = load_model(d, dev)
    keys = VALID_KEYS if split == "valid" else TEST_KEYS
    if split == "test":
        thr = runner.guard_test_run(d, allow_rerun)
    ds = ctx.dataset(keys)
    loader = make_loader(ds, batch_size)
    pred = runner.collect_predictions(model, loader, len(ds), d / f"pred_{split}", device=dev,
                                      size=int(ctx.pp["input"]["size"]))
    if split == "valid":
        log_f = d / "evaluations.jsonl"
        if log_f.exists() and any(json.loads(l).get("split") == "test" for l in log_f.read_text().splitlines() if l):
            raise PermissionError("thresholds are frozen: the test split was already evaluated for this run")
        thr = runner.select_and_freeze(pred, d / "threshold.json", n_bins=n_bins)
    overlap = list(ctx.cfg["patient_overlap"]["valid_test_ids"]) if split == "test" else []
    res = runner.evaluate_predictions(pred, thr, n_bins=n_bins, n_boot=n_boot, exclude_patients=overlap)
    if bench:
        res["engineering"] = runner.benchmark_latency(model, next(iter(loader))["image"], dev)
    cj = json.loads((d / "config.json").read_text())
    res["engineering"] = {**res.get("engineering", {}), "train_runtime_s": cj["runtime_s"],
                          "train_peak_vram_gb": cj["peak_vram_gb"], "checkpoint_mb": cj["checkpoint_mb"]}
    res["frozen_thresholds"] = thr
    res["split"], res["run_id"], res["model"] = split, d.name, cj["model"]
    (d / f"metrics_{split}.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    runner.log_evaluation(d, split, {"allow_rerun": allow_rerun})
    return res


def metrics_row(res: dict) -> dict:
    o = res["official"]
    pf, im = o["pixel_full_frame"], o["image"]
    eng = res.get("engineering", {})
    return {"run_id": res["run_id"], "split": res["split"], "model": res["model"],
            "image_auroc": im["image_auroc"], "image_aupr": im["image_aupr"], "pixel_auroc": pf["pixel_auroc"],
            "aupro": pf["aupro_fpr0.3"], "dice_val_thr": pf.get("dice_per_image", ""),
            "iou_val_thr": pf.get("iou_per_image", ""), "fpr": im.get("img_fpr", ""),
            "latency_ms": eng.get("latency_ms_mean", ""), "peak_vram_gb": eng.get("peak_vram_gb", ""), "status": "OK"}


# ------------------------------------------------------------------------------ selection
def select_model(run_dirs: list[str | Path]) -> dict:
    """Predeclared rule on VALIDATION metrics only.

    1. Gates: finite metrics; peak VRAM <= MAX_VRAM_GB and latency <= MAX_LATENCY_MS when measured.
    2. Localisation first: keep runs within SELECTION_MARGIN_AUPRO of the best AUPRO.
    3. Among those, best mean rank over (pixel AUROC, per-image Dice, image AUROC); ties -> lower latency.
    """
    cands = []
    for rd in run_dirs:
        f = Path(rd) / "metrics_valid.json"
        if not f.exists():
            raise FileNotFoundError(f"{f}: evaluate this run on 'valid' first")
        m = json.loads(f.read_text())
        pf, im, eng = m["official"]["pixel_full_frame"], m["official"]["image"], m.get("engineering", {})
        row = {"run": str(rd), "run_id": m["run_id"], "model": m["model"], "aupro": pf["aupro_fpr0.3"],
               "pixel_auroc": pf["pixel_auroc"], "dice": pf.get("dice_per_image", float("nan")),
               "image_auroc": im["image_auroc"], "latency_ms": eng.get("latency_ms_mean", 0.0),
               "vram": eng.get("peak_vram_gb") or 0.0}
        row["gate_ok"] = bool(np.isfinite([row["aupro"], row["pixel_auroc"], row["image_auroc"]]).all()
                              and row["vram"] <= MAX_VRAM_GB and row["latency_ms"] <= MAX_LATENCY_MS)
        cands.append(row)
    ok = [c for c in cands if c["gate_ok"]]
    if not ok:
        raise ValueError("no candidate passes the feasibility gates")
    best = max(c["aupro"] for c in ok)
    tied = [c for c in ok if c["aupro"] >= best - SELECTION_MARGIN_AUPRO]
    for k in ("pixel_auroc", "dice", "image_auroc"):
        order = sorted(tied, key=lambda c: -np.nan_to_num(c[k], nan=-1))
        for r, c in enumerate(order):
            c[f"rank_{k}"] = r + 1
    for c in tied:
        c["mean_rank"] = float(np.mean([c["rank_pixel_auroc"], c["rank_dice"], c["rank_image_auroc"]]))
    winner = sorted(tied, key=lambda c: (c["mean_rank"], c["latency_ms"]))[0]
    return {"rule": {"gates": {"max_vram_gb": MAX_VRAM_GB, "max_latency_ms": MAX_LATENCY_MS},
                     "aupro_margin": SELECTION_MARGIN_AUPRO,
                     "tie_break": "mean rank of pixel_auroc, per-image dice, image_auroc; then lower latency"},
            "selected": winner["run_id"], "selected_dir": winner["run"], "evaluated_on": "valid", "candidates": cands}


# ------------------------------------------------------------------------------ reporting
def _fmt(v, nd=4):
    return "NOT RUN" if v is None or v == "" else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))


def comparison_markdown(run_dirs: list[str | Path], splits=("valid", "test")) -> str:
    """Model-comparison tables built ONLY from metrics_<split>.json files on disk (else NOT RUN)."""
    out = ["# Model comparison (measured results only)", "",
           "Source: `artifacts/models/<run>/metrics_<split>.json`. A cell reads NOT RUN when no such file/value exists.",
           "Pixel metrics are full-frame; Dice/IoU are the mean over abnormal images at the validation-frozen pixel threshold;",
           "FPR is the image-level false-positive rate at the validation-frozen image threshold. Latency is per slice (batch 1).", ""]
    kinds = ["autoencoder", "padim", "patchcore"]
    for split in splits:
        out += [f"## {split}", "",
                "| Run | Model | Image AUROC | Image AUPR | Pixel AUROC | AUPRO | Dice | IoU | FPR | Latency (ms) | Peak VRAM (GB) |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        seen = set()
        for rd in run_dirs:
            f = Path(rd) / f"metrics_{split}.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text())
            o = m["official"]
            pf, im, eng = o["pixel_full_frame"], o["image"], m.get("engineering", {})
            seen.add(m["model"])
            out.append("| " + " | ".join([m["run_id"], m["model"], _fmt(im["image_auroc"]), _fmt(im["image_aupr"]),
                                          _fmt(pf["pixel_auroc"]), _fmt(pf["aupro_fpr0.3"]), _fmt(pf.get("dice_per_image")),
                                          _fmt(pf.get("iou_per_image")), _fmt(im.get("img_fpr")),
                                          _fmt(eng.get("latency_ms_mean"), 1), _fmt(eng.get("peak_vram_gb"), 2)]) + " |")
        for k in kinds:
            if k not in seen:
                out.append(f"| (none) | {k} | " + " | ".join(["NOT RUN"] * 9) + " |")
        out.append("")
    out += ["All outputs are research flags for qualified human review, not diagnosis."]
    return "\n".join(out) + "\n"


def pca_decision(baseline: dict, variant: dict, tol: float = 0.01, min_gain: float = 0.20) -> dict:
    """Predeclared PCA rule on VALIDATION metrics: keep a PCA variant only if it costs <= ``tol`` absolute in
    image AUROC, pixel AUROC and AUPRO versus the no-PCA baseline AND saves >= ``min_gain`` (20 %) in
    inference latency or stored model size. Inputs are dicts with those metric keys + latency_ms + model_mb."""
    drops = {k: baseline[k] - variant[k] for k in ("image_auroc", "pixel_auroc", "aupro")}
    gains = {"latency": 1 - variant["latency_ms"] / baseline["latency_ms"],
             "model_size": 1 - variant["model_mb"] / baseline["model_mb"]}
    quality_ok = all(d <= tol for d in drops.values())
    gain_ok = max(gains.values()) >= min_gain
    return {"keep": bool(quality_ok and gain_ok), "quality_ok": bool(quality_ok), "gain_ok": bool(gain_ok),
            "drops": drops, "gains": gains, "tol": tol, "min_gain": min_gain}
