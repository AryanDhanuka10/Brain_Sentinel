"""Frozen ImageNet ResNet feature extractor shared by PaDiM and PatchCore.

For a 240x240 input: layer2 -> (128, 30, 30), layer3 -> (256, 15, 15) (measured by scripts/check_env.py).
The input is the Day-3 network input: index/255 replicated to 3 channels, ImageNet-normalised. That is a
transfer-learning convention, not colour information.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

LAYER_CHANNELS = {"resnet18": {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}}


class ResNetFeatures(nn.Module):
    def __init__(self, backbone: str = "resnet18", layers: tuple[str, ...] = ("layer2", "layer3"),
                 pretrained: bool = True):
        super().__init__()
        if backbone != "resnet18":
            raise ValueError("only resnet18 is configured (8 GB VRAM plan)")
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        self.layers = tuple(layers)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.blocks = nn.ModuleDict({n: getattr(net, n) for n in ("layer1", "layer2", "layer3", "layer4")
                                     if n in self.layers or n < max(self.layers)})
        self.backbone, self.pretrained = backbone, pretrained
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # BatchNorm statistics must never update
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out, h = {}, self.stem(x)
        for name in sorted(self.blocks.keys()):
            h = self.blocks[name](h)
            if name in self.layers:
                out[name] = h
        return out

    @property
    def out_channels(self) -> int:
        return sum(LAYER_CHANNELS[self.backbone][n] for n in self.layers)


def embed_padim(feats: dict[str, torch.Tensor], layers: tuple[str, ...]) -> torch.Tensor:
    """PaDiM embedding: deeper layers bilinearly upsampled to the first layer's grid, channel-concat.
    (B, C, h, w). (The original unfolds; bilinear upsampling is the usual simplification.)"""
    ref = feats[layers[0]].shape[-2:]
    return torch.cat([feats[n] if feats[n].shape[-2:] == ref else
                      F.interpolate(feats[n], size=ref, mode="bilinear", align_corners=False)
                      for n in layers], dim=1)


def embed_patchcore(feats: dict[str, torch.Tensor], layers: tuple[str, ...], k: int = 3) -> torch.Tensor:
    """PatchCore local-neighbourhood aggregation (k x k average pool, stride 1) per layer, then as PaDiM."""
    agg = {n: F.avg_pool2d(feats[n], k, 1, k // 2) for n in layers}
    return embed_padim(agg, layers)
