"""Model C: PatchCore. Memory bank of normal-slice patch embeddings (greedy k-center coreset) + 1-NN distance.

Patch embedding: ResNet18 layer2+layer3, 3x3 local average aggregation, deeper layer upsampled (384-d).
Optional feature reducer (PCA/random) is applied before the bank is built; fitted on train patches only.
Patch score = distance to its nearest bank entry; map = bilinear upsample + Gaussian blur (sigma 4);
image score = max of the map. (The paper's re-weighting of the max score is not implemented.)
Coreset: greedy k-center on a fixed Gaussian random projection (seeded start point), run on ``device``.
"""
from __future__ import annotations

import time

import torch

from .common import AnomalyModel, upsample_smooth
from .features import ResNetFeatures, embed_patchcore
from .reducers import IdentityReducer, PCAReducer, RandomSelectReducer, reducer_from_state


def greedy_coreset(feats: torch.Tensor, n_select: int, proj_dim: int = 128, seed: int = 0,
                   device: str | torch.device = "cpu") -> torch.Tensor:
    """Indices of an approximate greedy k-center subset of ``feats`` (M,d)."""
    M = feats.shape[0]
    n_select = min(n_select, M)
    g = torch.Generator().manual_seed(seed)
    if feats.shape[1] > proj_dim:
        P = torch.randn(feats.shape[1], proj_dim, generator=g) / proj_dim ** 0.5
        z = (feats.to(device) @ P.to(device))
    else:
        z = feats.to(device)
    first = int(torch.randint(0, M, (1,), generator=g))
    sel = torch.empty(n_select, dtype=torch.long, device=z.device)
    sel[0] = first
    mind = ((z - z[first]) ** 2).sum(1)
    for i in range(1, n_select):
        j = int(torch.argmax(mind))
        sel[i] = j
        mind = torch.minimum(mind, ((z - z[j]) ** 2).sum(1))
    return sel.cpu()


class PatchCore(AnomalyModel):
    kind = "patchcore"

    def __init__(self, layers=("layer2", "layer3"), pretrained: bool = True, coreset_ratio: float = 0.01,
                 coreset_method: str = "greedy", proj_dim: int = 128, reduction: str = "none",
                 d_out: int = 128, pca_variance: float | None = None, sigma: float = 4.0, seed: int = 0,
                 input_size: int = 240, knn_chunk: int = 4096):
        super().__init__()
        if coreset_method not in ("greedy", "random"):
            raise ValueError("coreset_method must be greedy|random")
        self.layers, self.pretrained = tuple(layers), pretrained
        self.coreset_ratio, self.coreset_method, self.proj_dim = coreset_ratio, coreset_method, proj_dim
        self.reduction, self.d_out_req, self.pca_variance = reduction, d_out, pca_variance
        self.sigma, self.seed, self.input_size, self.knn_chunk = sigma, seed, input_size, knn_chunk
        self.backbone = ResNetFeatures("resnet18", self.layers, pretrained)
        self.d_emb = self.backbone.out_channels
        self.reducer = None
        self.register_buffer("bank", torch.zeros(0, 0))
        self.grid = None
        if reduction == "none":
            self.reducer = IdentityReducer(self.d_emb)
        elif reduction == "random":
            self.reducer = RandomSelectReducer(self.d_emb, d_out, seed)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        e = embed_patchcore(self.backbone(x), self.layers)
        self.grid = tuple(e.shape[-2:])
        return e.flatten(2).transpose(1, 2).contiguous()                     # (B,P,d_emb)

    @torch.no_grad()
    def fit(self, loader, device: str = "cpu", max_images: int | None = 2000, pca_fit_images: int = 500,
            log=print) -> dict:
        t0 = time.perf_counter()
        self.to(device)
        if self.reduction == "pca":
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
        chunks, seen = [], 0
        for b in loader:
            if max_images is not None and seen >= max_images:
                break
            x = b["image"].to(device)
            chunks.append(self.reducer.transform(self.embed(x)).reshape(-1, self.reducer.d_out).cpu())
            seen += x.shape[0]
        feats = torch.cat(chunks)
        M = feats.shape[0]
        n_sel = max(1, int(round(self.coreset_ratio * M)))
        t1 = time.perf_counter()
        if self.coreset_method == "greedy":
            idx = greedy_coreset(feats, n_sel, self.proj_dim, self.seed, device)
        else:
            idx = torch.randperm(M, generator=torch.Generator().manual_seed(self.seed))[:n_sel]
        self.bank = feats[idx].contiguous().to(device)
        return {"n_train_images": seen, "n_patches": M, "bank_size": int(self.bank.shape[0]),
                "d_used": int(self.bank.shape[1]), "coreset_seconds": time.perf_counter() - t1,
                "fit_seconds": time.perf_counter() - t0, "bank_mb": self.bank.numel() * 4 / 2**20}

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        if self.bank.numel() == 0:
            raise RuntimeError("PatchCore is not fitted")
        z = self.reducer.transform(self.embed(x))                            # (B,P,d)
        B, P, d = z.shape
        q = z.reshape(B * P, d)
        best = torch.empty(B * P, device=q.device)
        for a in range(0, B * P, self.knn_chunk):
            best[a:a + self.knn_chunk] = torch.cdist(q[a:a + self.knn_chunk], self.bank).min(1).values
        maps = upsample_smooth(best.view(B, *self.grid), self.input_size, self.sigma)
        return maps.float(), maps.flatten(1).max(1).values.float()

    def hparams(self) -> dict:
        return {"layers": list(self.layers), "pretrained": self.pretrained, "coreset_ratio": self.coreset_ratio,
                "coreset_method": self.coreset_method, "proj_dim": self.proj_dim, "reduction": self.reduction,
                "d_out": self.d_out_req, "pca_variance": self.pca_variance, "sigma": self.sigma,
                "seed": self.seed, "input_size": self.input_size}

    def state(self) -> dict:
        return {"bank": self.bank.cpu(), "grid": self.grid, "reducer": self.reducer.state(),
                "backbone": self.backbone.state_dict()}

    def load_state(self, s: dict) -> None:
        self.bank, self.grid = s["bank"].to(self.bank.device), tuple(s["grid"])
        self.reducer = reducer_from_state(s["reducer"])
        self.backbone.load_state_dict(s["backbone"])
