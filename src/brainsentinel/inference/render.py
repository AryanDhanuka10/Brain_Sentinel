"""Heatmap / overlay rendering. Display scale is fixed to the frozen pixel threshold (so colours are
comparable across slices) unless none exists, in which case it is per-image min-max (and labelled)."""
from __future__ import annotations

import io

import numpy as np
from matplotlib import colormaps
from PIL import Image


def _png(arr: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


def display_scale(amap: np.ndarray, pixel_threshold: float | None) -> tuple[float, float, str]:
    if pixel_threshold and pixel_threshold > 0:
        return 0.0, 2.0 * pixel_threshold, "fixed: 0 .. 2 x validation pixel threshold"
    return float(amap.min()), float(max(amap.max(), amap.min() + 1e-9)), "per-image min-max (no frozen threshold)"


def render_all(rgb_original: np.ndarray, amap: np.ndarray, pixel_threshold: float | None, alpha: float = 0.45) -> dict:
    """-> {'original','heatmap','overlay'} PNG bytes + the display-scale description."""
    lo, hi, desc = display_scale(amap, pixel_threshold)
    t = np.clip((amap - lo) / (hi - lo), 0, 1)
    heat = (colormaps["inferno"](t)[..., :3] * 255).astype(np.uint8)
    base = rgb_original.astype(np.float32)
    w = (alpha * t)[..., None]
    overlay = (base * (1 - w) + heat.astype(np.float32) * w).clip(0, 255).astype(np.uint8)
    return {"original": _png(rgb_original), "heatmap": _png(heat), "overlay": _png(overlay), "display_scale": desc}
