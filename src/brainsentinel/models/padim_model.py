"""Model B: PaDiM. Per-position multivariate Gaussian over frozen ResNet18 embeddings of normal slices.

Anomaly score at position p = Mahalanobis distance sqrt((x-mu_p)^T Sigma_p^-1 (x-mu_p)).
Map = bilinear upsample to 240 + Gaussian blur (sigma 4, as in the paper); image score = max of the map.
Mean/covariance come from streamed float64 moments computed ONLY from the loader passed to ``fit``.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from .common import AnomalyModel, upsample_smooth
from .features import ResNetFeatures, embed_padim
from .reducers import IdentityReducer, PCAReducer, RandomSelectReducer, reducer_from_state


class PaDiM(AnomalyModel):
    kind = "padim"

    def __init__(self, layers=("layer2", "layer3"), pretrained: bool = True, reduction: str = "random",
                 d_out: int = 100, pca_variance: float | None = None, reg_eps: float = 0.01,
                 sigma: float = 4.0, seed: int = 0, input_size: int = 240):
        super().__init__()
        self.layers, self.pretrained = tuple(layers), pretrained
        self.reduction, self.d_out_req, self.pca_variance = reduction, d_out, pca_variance
        self.reg_eps, self.sigma, self.seed, self.input_size = reg_eps, sigma, seed, input_size
        self.backbone = ResNetFeatures("resnet18", self.layers, pretrained)
        self.d_emb = self.backbone.out_channels
        self.reducer = None
        self.register_buffer("mean", torch.zeros(0))
        self.register_buffer("inv_cov", torch.zeros(0))
        self.grid = None
        if reduction == "none":
            self.reducer = IdentityReducer(self.d_emb)
        elif reduction == "random":
            self.reducer = RandomSelectReducer(self.d_emb, d_out, seed)

    # ---- embedding ---------------------------------------------------------------------------
    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B,P,d_emb) before reduction."""
        e = embed_padim(self.backbone(x), self.layers)
        self.grid = tuple(e.shape[-2:])
        return e.flatten(2).transpose(1, 2).contiguous()

    @torch.no_grad()
    def fit(self, loader, device: str = "cpu", max_images: int | None = None, pca_fit_images: int = 500,
            log=print) -> dict:
        t0 = time.perf_counter()
        self.to(device)
        seen = 0
        if self.reduction == "pca":                               # pass 1: PCA on train embeddings only
            self.reducer = PCAReducer(self.d_emb, variance=self.pca_variance)

            def gen():
                n = 0
                for b in loader:
                    yield self.embed(b["image"].to(device))
                    n += b["image"].shape[0]
                    if n >= pca_fit_images:
                        return
            self.reducer.fit(gen(), device=device)
            log(f"PCA: {self.d_emb} -> {self.reducer.d_out} dims, explained {self.reducer.explained:.4f}")
        s1 = s2 = shift = None
        N = 0
        for b in loader:                                          # pass 2: Gaussian moments
            x = b["image"].to(device)
            if max_images is not None and seen >= max_images:
                break
            z = self.reducer.transform(self.embed(x)).to(torch.float64)         # (B,P,d)
            if shift is None:
                shift = z.mean(0)                                                # (P,d)
                s1 = torch.zeros_like(shift)
                s2 = torch.zeros(z.shape[1], z.shape[2], z.shape[2], dtype=torch.float64, device=z.device)
            c = z - shift
            s1 += c.sum(0)
            s2 += torch.einsum("bpd,bpe->pde", c, c)
            N += z.shape[0]
            seen += x.shape[0]
        if N < 2:
            raise ValueError("PaDiM needs at least 2 training images")
        m = s1 / N
        cov = (s2 - N * torch.einsum("pd,pe->pde", m, m)) / (N - 1)
        d = cov.shape[-1]
        cov = cov + self.reg_eps * torch.eye(d, dtype=cov.dtype, device=cov.device)
        self.mean = (shift + m).float()
        self.inv_cov = torch.linalg.inv(cov).float()
        if not (torch.isfinite(self.mean).all() and torch.isfinite(self.inv_cov).all()):
            raise FloatingPointError("non-finite PaDiM statistics")
        return {"n_train_images": N, "d_embedding": self.d_emb, "d_used": d, "grid": self.grid,
                "fit_seconds": time.perf_counter() - t0,
                "stats_mb": (self.mean.numel() + self.inv_cov.numel()) * 4 / 2**20}

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        if self.mean.numel() == 0:
            raise RuntimeError("PaDiM is not fitted")
        z = self.reducer.transform(self.embed(x))                                # (B,P,d)
        delta = z - self.mean
        d2 = torch.einsum("bpd,pde,bpe->bp", delta, self.inv_cov, delta).clamp_min(0)
        grid = d2.sqrt().view(x.shape[0], *self.grid)
        maps = upsample_smooth(grid, self.input_size, self.sigma)
        return maps.float(), maps.flatten(1).max(1).values.float()

    def hparams(self) -> dict:
        return {"layers": list(self.layers), "pretrained": self.pretrained, "reduction": self.reduction,
                "d_out": self.d_out_req, "pca_variance": self.pca_variance, "reg_eps": self.reg_eps,
                "sigma": self.sigma, "seed": self.seed, "input_size": self.input_size}

    def state(self) -> dict:
        return {"mean": self.mean.cpu(), "inv_cov": self.inv_cov.cpu(), "grid": self.grid,
                "reducer": self.reducer.state(), "backbone": self.backbone.state_dict()}

    def load_state(self, s: dict) -> None:
        self.mean, self.inv_cov, self.grid = s["mean"].to(self.mean.device), s["inv_cov"], tuple(s["grid"])
        self.reducer = reducer_from_state(s["reducer"])
        self.backbone.load_state_dict(s["backbone"])
