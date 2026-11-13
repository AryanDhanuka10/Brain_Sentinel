"""Predictor: a trained run directory -> scores and anomaly maps for validated slices (no LLM involved)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data.preprocessing import to_network_input
from ..models import load_model
from ..pipeline import load_palette


class Predictor:
    def __init__(self, run_dir: str | Path, device: str = "cpu", palette_path: str | Path | None = None):
        d = Path(run_dir)
        self.run_dir, self.device = d, device
        self.cfg = json.loads((d / "config.json").read_text())
        self.model = load_model(d, device)
        self.mean, self.std = tuple(self.cfg["normalize"]["mean"]), tuple(self.cfg["normalize"]["std"])
        self.palette = load_palette(palette_path or d / "palette.json")
        tf = d / "threshold.json"
        self.thresholds = json.loads(tf.read_text()) if tf.exists() else None
        if self.thresholds and self.thresholds.get("selected_on") != "valid":
            raise ValueError(f"{tf}: thresholds not selected on validation")
        self.model_id = self.cfg["run_id"]
        self.size = int(self.cfg["input_size"])

    @property
    def info(self) -> dict:
        c = self.cfg
        mv = self.run_dir / "metrics_valid.json"
        return {"model_id": self.model_id, "kind": c["model"], "hparams": c["hparams"], "trained_on": c["trained_on"],
                "seed": c["seed"], "normalize_mode": c["normalize"]["mode"], "input_size": self.size,
                "checkpoint_mb": c.get("checkpoint_mb"), "train_runtime_s": c.get("runtime_s"), "device": self.device,
                "thresholds": self.thresholds, "validation_metrics_available": mv.exists(),
                "validation_summary": (lambda m: {"image_auroc": m["official"]["image"]["image_auroc"],
                                                  "pixel_auroc": m["official"]["pixel_full_frame"]["pixel_auroc"],
                                                  "aupro": m["official"]["pixel_full_frame"]["aupro_fpr0.3"],
                                                  "latency_ms_mean": m.get("engineering", {}).get("latency_ms_mean")})(
                    json.loads(mv.read_text())) if mv.exists() else None}

    def _tensor(self, idxs: list[np.ndarray]) -> torch.Tensor:
        return torch.from_numpy(np.stack([to_network_input(i, self.mean, self.std) for i in idxs])).to(self.device)

    @torch.no_grad()
    def predict(self, idxs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
        """list of (240,240) uint8 index images -> maps (B,H,W) float32, scores (B,), latency_ms for the batch."""
        x = self._tensor(idxs)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        maps, scores = self.model.predict(x)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        m, s = maps.float().cpu().numpy(), scores.float().cpu().numpy()
        if not (np.isfinite(m).all() and np.isfinite(s).all()):
            raise FloatingPointError("model produced non-finite output")
        return m, s, ms


def resolve_model_dir(repo: Path) -> Path | None:
    """BRAINSENTINEL_MODEL_DIR, else artifacts/selected_model.json -> selected_dir. None when nothing is trained."""
    import os
    env = os.environ.get("BRAINSENTINEL_MODEL_DIR")
    if env:
        return Path(env)
    f = repo / "artifacts" / "selected_model.json"
    if f.exists():
        p = Path(json.loads(f.read_text())["selected_dir"])
        return p if p.is_absolute() else repo / p
    return None
