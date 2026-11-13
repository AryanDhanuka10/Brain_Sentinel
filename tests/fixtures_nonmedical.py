"""NON-MEDICAL numeric fixtures for code-path tests ONLY.

Smooth random fields with an optional bright square. They are not MRI, are never written to disk or the
dataset directory, and no reported metric is ever computed from them. Real-data evaluation runs only
through scripts/train.py + scripts/evaluate.py on the user's local BMAD data.
"""
from __future__ import annotations

import numpy as np
import torch


def field(n: int, size: int, seed: int, square: tuple[int, int, int] | None = None, amp: float = 3.0) -> torch.Tensor:
    """(n,3,size,size) float32. square=(y, x, side) adds a bright square to every image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    out = []
    for _ in range(n):
        a, b, c = rng.normal(0, 0.3, 3)
        g = 0.5 * np.sin(2 * np.pi * (xx + a)) + 0.5 * np.cos(2 * np.pi * (yy + b)) + c
        g = g + rng.normal(0, 0.05, (size, size))
        if square is not None:
            y, x, s = square
            g[y:y + s, x:x + s] += amp
        out.append(np.repeat(g[None], 3, axis=0))
    return torch.from_numpy(np.stack(out).astype(np.float32))


def batches(x: torch.Tensor, bs: int = 4):
    return [{"image": x[i:i + bs]} for i in range(0, len(x), bs)]
