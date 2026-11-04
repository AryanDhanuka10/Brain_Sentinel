"""Recover 8-bit intensity indices from BMAD Brain RGB PNGs (palette is a data fact, built on train only)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .checks import list_images


def pack_rgb(rgb: np.ndarray) -> np.ndarray:
    """(...,3+) uint8 -> (...) uint32 key = R<<16 | G<<8 | B."""
    a = rgb[..., :3].astype(np.uint32)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


def load_rgb(path: Path) -> np.ndarray:
    """Load a PNG as (H, W, 3) uint8. Raises if not RGBA with alpha == 255."""
    arr = np.asarray(Image.open(path).convert("RGBA"))
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"{path}: unexpected shape {arr.shape}")
    if not (arr[..., 3] == 255).all():
        raise ValueError(f"{path}: alpha is not 255 everywhere")
    return arr[..., :3].copy()


@dataclass(frozen=True, eq=False)
class Palette:
    """Sorted unique colours; index = position in the sorted order."""

    colors: np.ndarray  # (K, 3) uint8

    @classmethod
    def from_keys(cls, keys: np.ndarray) -> "Palette":
        k = np.unique(np.asarray(keys, dtype=np.uint32))
        if not 1 <= len(k) <= 256:
            raise ValueError(f"need 1..256 colours for uint8 index, got {len(k)}")
        colors = np.stack([(k >> 16) & 255, (k >> 8) & 255, k & 255], axis=1).astype(np.uint8)
        return cls(colors)

    @property
    def keys(self) -> np.ndarray:
        return pack_rgb(self.colors)

    def is_monotone(self) -> bool:
        """True if R, G and B are all non-decreasing with index (index is then an intensity order)."""
        return bool((np.diff(self.colors.astype(np.int16), axis=0) >= 0).all())

    def encode(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 -> (H, W) uint8 index. Raises on any colour not in the palette."""
        keys = pack_rgb(rgb)
        pos = np.searchsorted(self.keys, keys)
        pos = np.clip(pos, 0, len(self.keys) - 1)
        ok = self.keys[pos] == keys
        if not ok.all():
            raise ValueError(f"{int((~ok).sum())} pixels have colours outside the palette")
        return pos.astype(np.uint8)

    def decode(self, idx: np.ndarray) -> np.ndarray:
        return self.colors[idx]


def corner_bg(idx: np.ndarray) -> int:
    """Background index estimated from the four corners. Raises if they differ."""
    c = {int(idx[0, 0]), int(idx[0, -1]), int(idx[-1, 0]), int(idx[-1, -1])}
    if len(c) != 1:
        raise ValueError(f"corner pixels differ: {sorted(c)}")
    return c.pop()


def brain_mask(idx: np.ndarray) -> np.ndarray:
    """HYPOTHESIS: brain = pixels brighter than the per-image corner background."""
    return idx > corner_bg(idx)


def train_palette(cfg: dict) -> Palette:
    """Palette from train/good only (never valid/test)."""
    seen = np.zeros(1 << 24, dtype=bool)
    for p in list_images(cfg, "train/good"):
        seen[pack_rgb(load_rgb(p)).ravel()] = True
    return Palette.from_keys(np.flatnonzero(seen))