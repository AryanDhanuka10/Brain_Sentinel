"""Shared model utilities: Gaussian smoothing, map upsampling, base class."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def gaussian_kernel1d(sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    radius = max(1, int(4.0 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (B,1,H,W) with reflect padding. sigma <= 0 -> identity."""
    if sigma <= 0:
        return x
    k = gaussian_kernel1d(sigma, x.device, x.dtype)
    r = (len(k) - 1) // 2
    kh, kv = k.view(1, 1, 1, -1), k.view(1, 1, -1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kh)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), kv)


def upsample_smooth(grid: torch.Tensor, size: int, sigma: float) -> torch.Tensor:
    """(B,h,w) patch-score grid -> (B,size,size) smoothed anomaly map (bilinear, then Gaussian)."""
    m = F.interpolate(grid.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False)
    return gaussian_blur(m, sigma).squeeze(1)


def topk_mean(maps: torch.Tensor, frac: float) -> torch.Tensor:
    """Mean of the top ``frac`` of pixels of each (B,H,W) map (predeclared image-score rule)."""
    flat = maps.flatten(1)
    k = max(1, int(math.ceil(frac * flat.shape[1])))
    return flat.topk(k, dim=1).values.mean(1)


class AnomalyModel(nn.Module):
    """Interface every detector implements. Models are trained on normal slices only.

    predict(x): x (B,3,H,W) float32 (Day-3 network input) -> (maps (B,H,W) float32, scores (B,)).
    Higher = more anomalous. Maps are at the input resolution so they align pixel-for-pixel with masks.
    """

    kind = "base"
    input_size = 240

    def fit(self, loader, **kw) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    def hparams(self) -> dict:  # pragma: no cover
        raise NotImplementedError
