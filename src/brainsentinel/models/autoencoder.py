"""Model A: convolutional autoencoder (reconstruction baseline). Trained on normal slices only.

Anomaly map  = Gaussian-smoothed |x - x_hat| in normalised-intensity units (full frame).
Image score = mean of the top ``topk_frac`` of that map (predeclared; default top 1% of pixels).
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn

from .common import AnomalyModel, gaussian_blur, topk_mean


def _down(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True))


def _up(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class ConvAutoencoder(AnomalyModel):
    kind = "autoencoder"

    def __init__(self, width: int = 32, latent_channels: int = 16, sigma: float = 2.0,
                 topk_frac: float = 0.01, train_noise_std: float = 0.0, input_size: int = 240):
        super().__init__()
        w = width
        if input_size % 16:
            raise ValueError("input_size must be divisible by 16")
        self.width, self.latent_channels, self.sigma = width, latent_channels, sigma
        self.topk_frac, self.train_noise_std, self.input_size = topk_frac, train_noise_std, input_size
        self.enc = nn.Sequential(_down(1, w), _down(w, 2 * w), _down(2 * w, 4 * w), _down(4 * w, 8 * w),
                                 nn.Conv2d(8 * w, latent_channels, 1))               # 240 -> 15x15xC
        self.dec = nn.Sequential(nn.Conv2d(latent_channels, 8 * w, 1), nn.ReLU(True),
                                 _up(8 * w, 4 * w), _up(4 * w, 2 * w), _up(2 * w, w),
                                 nn.ConvTranspose2d(w, 1, 4, 2, 1))

    @staticmethod
    def to_gray(x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) Day-3 input -> (B,1,H,W). Channels are affine copies of one gray image."""
        return x.mean(1, keepdim=True)

    def reconstruct(self, g: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(g))

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        g = self.to_gray(x)
        inp = g + self.train_noise_std * torch.randn_like(g) if (self.training and self.train_noise_std > 0) else g
        return F.mse_loss(self.reconstruct(inp), g)

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        was = self.training
        self.eval()
        g = self.to_gray(x)
        res = (g - self.reconstruct(g)).abs()
        maps = gaussian_blur(res, self.sigma).squeeze(1)
        self.train(was)
        return maps.float(), topk_mean(maps, self.topk_frac).float()

    @torch.no_grad()
    def val_loss(self, loader, device) -> float:
        self.eval()
        tot, n = 0.0, 0
        for b in loader:
            x = b["image"].to(device)
            tot += float(F.mse_loss(self.reconstruct(self.to_gray(x)), self.to_gray(x), reduction="sum"))
            n += x[:, :1].numel()
        return tot / n

    def fit(self, loader, val_loader=None, epochs: int = 30, lr: float = 1e-3, weight_decay: float = 1e-5,
            patience: int = 5, device: str = "cpu", log=print, max_batches: int | None = None) -> dict:
        """Train on normal slices. ``val_loader`` must contain NORMAL validation slices only (valid/good):
        it drives early stopping; abnormal slices and masks are never seen here."""
        self.to(device)
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        best, best_state, bad, hist = float("inf"), None, 0, []
        t0 = time.perf_counter()
        for ep in range(1, epochs + 1):
            self.train()
            tot, n = 0.0, 0
            for bi, b in enumerate(loader):
                if max_batches is not None and bi >= max_batches:
                    break
                x = b["image"].to(device)
                loss = self.loss(x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot, n = tot + float(loss.detach()) * x.shape[0], n + x.shape[0]
            tr = tot / max(n, 1)
            va = self.val_loss(val_loader, device) if val_loader is not None else float("nan")
            hist.append({"epoch": ep, "train_mse": tr, "val_normal_mse": va})
            log(f"epoch {ep:3d} train_mse {tr:.6f} val_normal_mse {va:.6f}")
            crit = va if val_loader is not None else tr
            if crit < best - 1e-7:
                best, bad = crit, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    log(f"early stop at epoch {ep}")
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        return {"history": hist, "best_criterion": best, "epochs_run": len(hist),
                "fit_seconds": time.perf_counter() - t0}

    def hparams(self) -> dict:
        return {"width": self.width, "latent_channels": self.latent_channels, "sigma": self.sigma,
                "topk_frac": self.topk_frac, "train_noise_std": self.train_noise_std, "input_size": self.input_size}

    def state(self) -> dict:
        return {"state_dict": self.state_dict()}

    def load_state(self, s: dict) -> None:
        self.load_state_dict(s["state_dict"])
