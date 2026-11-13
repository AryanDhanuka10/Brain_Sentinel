"""Model registry and run-directory (de)serialisation."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .autoencoder import ConvAutoencoder
from .common import AnomalyModel
from .padim_model import PaDiM
from .patchcore_model import PatchCore

REGISTRY = {"autoencoder": ConvAutoencoder, "padim": PaDiM, "patchcore": PatchCore}


def build_model(kind: str, hparams: dict | None = None) -> AnomalyModel:
    if kind not in REGISTRY:
        raise ValueError(f"unknown model {kind!r}; choose from {sorted(REGISTRY)}")
    hp = dict(hparams or {})
    for k in ("layers",):
        if k in hp:
            hp[k] = tuple(hp[k])
    return REGISTRY[kind](**hp)


def save_model(model: AnomalyModel, run_dir: str | Path) -> int:
    """Write model.pt and model.json (kind + hparams). Returns checkpoint size in bytes."""
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state(), d / "model.pt")
    (d / "model.json").write_text(json.dumps({"kind": model.kind, "hparams": model.hparams()}, indent=2) + "\n")
    return (d / "model.pt").stat().st_size


def load_model(run_dir: str | Path, device: str = "cpu") -> AnomalyModel:
    d = Path(run_dir)
    meta = json.loads((d / "model.json").read_text())
    hp = dict(meta["hparams"])
    if meta["kind"] != "autoencoder":
        hp["pretrained"] = False        # weights come from the checkpoint, never from a download
    m = build_model(meta["kind"], hp)
    m.load_state(torch.load(d / "model.pt", map_location="cpu", weights_only=False))
    return m.to(device).eval()
