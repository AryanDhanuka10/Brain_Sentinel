"""Threshold selection. Thresholds are chosen on the VALIDATION split only, then frozen to JSON.

Nothing in this module may be called with test scores to *choose* a threshold; ``freeze`` records the
split name and refuses anything but ``valid``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from .metrics import PixelStats


def select_image_threshold(scores: np.ndarray, labels: np.ndarray, rule: str = "max_f1",
                           target_sensitivity: float = 0.95) -> float:
    """Image-level threshold (flag if score >= thr).

    rules: ``max_f1`` (default), ``youden`` (max sens+spec-1), ``sens_at_least`` (highest threshold with
    sensitivity >= target_sensitivity). Ties resolve to the HIGHER threshold (fewer false flags).
    """
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("need both classes to select an image threshold")
    if rule == "max_f1":
        p, r, t = precision_recall_curve(y, s)
        f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
        best = np.flatnonzero(f1 >= f1.max() - 1e-12)
        return float(t[best[-1]])
    fpr, tpr, t = roc_curve(y, s)
    ok = np.isfinite(t)
    fpr, tpr, t = fpr[ok], tpr[ok], t[ok]
    if rule == "youden":
        j = tpr - fpr
        best = np.flatnonzero(j >= j.max() - 1e-12)
        return float(t[best[0]])                       # roc_curve thresholds are descending: first = highest
    if rule == "sens_at_least":
        idx = np.flatnonzero(tpr >= target_sensitivity)
        if len(idx) == 0:
            raise ValueError("target sensitivity unreachable")
        return float(t[idx[0]])
    raise ValueError(f"unknown rule {rule!r}")


def select_pixel_threshold(stats: PixelStats, rule: str = "max_dice") -> float:
    """Pixel threshold from validation histograms: maximise pooled Dice (``max_dice``) or IoU (``max_iou``)."""
    thr, dice, iou = stats.dice_iou_curve()
    curve = {"max_dice": dice, "max_iou": iou}.get(rule)
    if curve is None:
        raise ValueError(f"unknown rule {rule!r}")
    return float(thr[int(np.argmax(curve))])


def freeze(path: str | Path, *, split: str, image_threshold: float, image_rule: str,
           pixel_threshold: float, pixel_rule: str, n_images: int, extra: dict | None = None) -> dict:
    """Write the frozen thresholds with provenance. Only validation-derived thresholds may be frozen."""
    if split != "valid":
        raise ValueError(f"thresholds may only be selected on 'valid', got {split!r}")
    rec = {"selected_on": split, "n_images": int(n_images),
           "image_threshold": float(image_threshold), "image_rule": image_rule,
           "pixel_threshold": float(pixel_threshold), "pixel_rule": pixel_rule, **(extra or {})}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def load_frozen(path: str | Path) -> dict:
    rec = json.loads(Path(path).read_text())
    if rec.get("selected_on") != "valid":
        raise ValueError(f"{path}: thresholds not selected on validation")
    return rec
