"""Feature-dimension reducers fitted on TRAIN-normal embeddings only: none / random selection / PCA."""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch


class IdentityReducer:
    kind = "none"

    def __init__(self, d_in: int):
        self.d_in, self.d_out = d_in, d_in

    def transform(self, x: torch.Tensor) -> torch.Tensor:  # x (..., d_in)
        return x

    def state(self) -> dict:
        return {"kind": "none", "d_in": self.d_in}


class RandomSelectReducer:
    """Seeded random subset of channels (the PaDiM paper's dimensionality reduction). Needs no fitting."""
    kind = "random"

    def __init__(self, d_in: int, d_out: int, seed: int = 0, index: torch.Tensor | None = None):
        if not 1 <= d_out <= d_in:
            raise ValueError(f"d_out must be in [1, {d_in}], got {d_out}")
        self.d_in, self.d_out = d_in, d_out
        self.index = index if index is not None else torch.from_numpy(
            np.sort(np.random.default_rng(seed).choice(d_in, d_out, replace=False)))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(-1, self.index.to(x.device))

    def state(self) -> dict:
        return {"kind": "random", "d_in": self.d_in, "d_out": self.d_out, "index": self.index.cpu()}


class PCAReducer:
    """Exact PCA from streamed first/second moments (float64), so it never holds all vectors in memory.

    ``n_components`` (int) or ``variance`` (0<v<=1: smallest k with cumulative explained variance >= v).
    Fitted ONLY on vectors passed to ``fit`` (the caller passes train/good embeddings). Not whitened.
    """
    kind = "pca"

    def __init__(self, d_in: int, n_components: int | None = None, variance: float | None = None):
        if (n_components is None) == (variance is None):
            raise ValueError("give exactly one of n_components / variance")
        if variance is not None and not 0 < variance <= 1:
            raise ValueError("variance must be in (0, 1]")
        self.d_in, self.n_components_req, self.variance_req = d_in, n_components, variance
        self.mean = self.components = self.explained_variance_ratio = None
        self.d_out = n_components or d_in
        self.fit_seconds = 0.0
        self.n_fit_vectors = 0

    def fit(self, batches: Iterable[torch.Tensor], device: str | torch.device = "cpu") -> "PCAReducer":
        t0 = time.perf_counter()
        s1 = torch.zeros(self.d_in, dtype=torch.float64, device=device)
        s2 = torch.zeros(self.d_in, self.d_in, dtype=torch.float64, device=device)
        n, shift = 0, None
        for b in batches:
            b = b.reshape(-1, self.d_in).to(device=device, dtype=torch.float64)
            if shift is None:
                shift = b.mean(0)                       # shifting avoids catastrophic cancellation
            c = b - shift
            s1 += c.sum(0)
            s2 += c.T @ c
            n += b.shape[0]
        if n < 2:
            raise ValueError("PCA needs at least 2 vectors")
        m = s1 / n
        cov = (s2 - n * torch.outer(m, m)) / (n - 1)
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)
        evals, evecs = evals[order].clamp_min(0), evecs[:, order]
        ratio = evals / evals.sum().clamp_min(1e-30)
        k = self.n_components_req or int(torch.searchsorted(torch.cumsum(ratio, 0),
                                                            torch.tensor(self.variance_req - 1e-12,
                                                                         dtype=ratio.dtype, device=ratio.device)).item()) + 1
        k = max(1, min(k, self.d_in))
        self.mean = (shift + m).float().cpu()
        self.components = evecs[:, :k].T.float().cpu().contiguous()      # (k, d_in)
        self.explained_variance_ratio = ratio.float().cpu()
        self.d_out, self.n_fit_vectors = k, n
        self.fit_seconds = time.perf_counter() - t0
        return self

    @property
    def explained(self) -> float:
        return float(self.explained_variance_ratio[: self.d_out].sum())

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.components is None:
            raise RuntimeError("PCAReducer not fitted")
        return (x - self.mean.to(x.device)) @ self.components.to(x.device).T

    def state(self) -> dict:
        return {"kind": "pca", "d_in": self.d_in, "d_out": self.d_out, "mean": self.mean,
                "components": self.components, "explained_variance_ratio": self.explained_variance_ratio,
                "n_fit_vectors": self.n_fit_vectors, "fit_seconds": self.fit_seconds}


def reducer_from_state(s: dict):
    if s["kind"] == "none":
        return IdentityReducer(s["d_in"])
    if s["kind"] == "random":
        return RandomSelectReducer(s["d_in"], s["d_out"], index=s["index"])
    r = PCAReducer(s["d_in"], n_components=s["d_out"])
    r.mean, r.components = s["mean"], s["components"]
    r.explained_variance_ratio, r.d_out = s["explained_variance_ratio"], s["d_out"]
    r.n_fit_vectors, r.fit_seconds = s.get("n_fit_vectors", 0), s.get("fit_seconds", 0.0)
    return r
