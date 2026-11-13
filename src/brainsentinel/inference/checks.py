"""Deterministic input checks: decode/validate an upload, domain conformance, image quality.

DOMAIN CHECK = conformance to the training input format, NOT a modality classifier. The detector was
trained on BMAD Brain slices: 240x240 RGBA PNG whose colours all come from one 256-colour palette
(a fixed intensity -> colour map), uniform corner background, a brain-sized foreground. An upload that
does not conform (a photo, a CT, a screenshot, a differently-rendered MRI, an RGB export) is REJECTED
rather than pushed through the brain detector. A conforming image of another sequence (e.g. T1 rendered
through the same palette) cannot be detected by this check; see docs/KNOWN_LIMITATIONS.md.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy import ndimage

from ..data.palette import MAX_BG, Palette, brain_mask, corner_bg, pack_rgb

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
# Heuristic limits (documented in docs/AGENTIC_WORKFLOW.md). Train brain fraction is ~0.27-0.30 (Day 2).
QUALITY_LIMITS = {"brain_fraction_fail": (0.05, 0.60), "brain_fraction_warn": (0.12, 0.45),
                  "saturated_share_warn": 0.02, "laplacian_var_fail": 1.0, "intensity_std_fail": 2.0}


class InputRejected(ValueError):
    """The upload cannot be processed. ``code`` is a stable machine-readable reason."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class DecodedSlice:
    idx: np.ndarray                 # (240,240) uint8, corner background subtracted (training convention)
    brain: np.ndarray               # (240,240) bool
    bg: int
    sha256: str
    n_bytes: int
    domain: dict = field(default_factory=dict)


def decode_upload(data: bytes, pal: Palette, size: int = 240) -> DecodedSlice:
    """bytes -> DecodedSlice or InputRejected. Mirrors data.preprocessing.load_index_image."""
    if not data:
        raise InputRejected("empty_file", "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InputRejected("file_too_large", f"File exceeds {MAX_UPLOAD_BYTES // 2**20} MB.")
    if not data.startswith(PNG_MAGIC):
        raise InputRejected("not_png", "Only PNG slices are accepted in this prototype.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            mode, dims = im.mode, im.size
            arr = np.asarray(im.convert("RGBA"))
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:  # truncated / corrupt
        raise InputRejected("corrupt_image", f"The PNG could not be decoded: {e}") from e
    dom = {"format": "PNG", "mode": mode, "width": dims[0], "height": dims[1]}
    if arr.shape[:2] != (size, size):
        raise InputRejected("wrong_size", f"Expected {size}x{size} pixels, got {dims[0]}x{dims[1]}.")
    if mode != "RGBA" or not (arr[..., 3] == 255).all():
        raise InputRejected("unsupported_encoding",
                            "Expected an opaque RGBA slice in the BMAD palette encoding this model was trained on "
                            f"(got mode {mode}). Other encodings and modalities are not supported.")
    rgb = arr[..., :3]
    keys = pack_rgb(rgb)
    pos = np.clip(np.searchsorted(pal.keys, keys), 0, len(pal.keys) - 1)
    off = int((pal.keys[pos] != keys).sum())
    dom["pixels_outside_training_palette"] = off
    if off:
        raise InputRejected("unsupported_encoding",
                            f"{off} pixels use colours outside the training palette; this is not a BMAD-encoded slice.")
    idx = pos.astype(np.uint8)
    try:
        bg = corner_bg(idx)
    except ValueError as e:
        raise InputRejected("unsupported_background", str(e)) from e
    brain = brain_mask(idx)
    dom.update({"corner_background_index": bg, "palette_conformant": True})
    out = np.clip(idx.astype(np.int16) - bg, 0, 255).astype(np.uint8)
    return DecodedSlice(out, brain, bg, hashlib.sha256(data).hexdigest(), len(data), dom)


def assess_quality(s: DecodedSlice) -> dict:
    """Deterministic quality measures + flags. level: pass | warn | fail. Never diagnostic."""
    idx, brain = s.idx.astype(np.float32), s.brain
    frac = float(brain.mean())
    vals = idx[brain]
    inner = ndimage.binary_erosion(brain, iterations=3)
    lap = ndimage.laplace(idx)
    lap_var = float(lap[inner].var()) if inner.any() else 0.0
    m = {"brain_fraction": frac, "intensity_mean": float(vals.mean()) if vals.size else 0.0,
         "intensity_std": float(vals.std()) if vals.size else 0.0,
         "saturated_share": float((s.idx[brain] == 255).mean()) if vals.size else 0.0,
         "laplacian_variance_interior": lap_var, "corner_background_index": s.bg}
    L, flags = QUALITY_LIMITS, []
    lo, hi = L["brain_fraction_fail"]
    if not lo <= frac <= hi:
        flags.append({"code": "brain_fraction_out_of_range", "level": "fail",
                      "detail": f"foreground fraction {frac:.3f} outside [{lo}, {hi}]"})
    elif not L["brain_fraction_warn"][0] <= frac <= L["brain_fraction_warn"][1]:
        flags.append({"code": "brain_fraction_unusual", "level": "warn", "detail": f"foreground fraction {frac:.3f}"})
    if m["intensity_std"] < L["intensity_std_fail"]:
        flags.append({"code": "near_uniform_image", "level": "fail", "detail": f"intensity std {m['intensity_std']:.2f}"})
    if lap_var < L["laplacian_var_fail"] and not any(f["code"] == "near_uniform_image" for f in flags):
        flags.append({"code": "no_texture_or_blank", "level": "fail", "detail": f"interior Laplacian variance {lap_var:.2f}"})
    if m["saturated_share"] > L["saturated_share_warn"]:
        flags.append({"code": "saturation", "level": "warn", "detail": f"{m['saturated_share']:.3f} of foreground at max intensity"})
    if s.bg > 0:
        flags.append({"code": "offset_background", "level": "warn",
                      "detail": f"background index {s.bg} (seen in a few training slices; corrected as in training)"})
    level = "fail" if any(f["level"] == "fail" for f in flags) else ("warn" if flags else "pass")
    return {"level": level, "metrics": m, "flags": flags, "limits": L}
