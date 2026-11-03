"""Dataset-contract helpers for the BMAD Brain subset (no model code)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

NAME_RE = re.compile(r"^(\d+)_(\d+)\.png$")


def load_config(path: str | Path = "configs/data.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def parse_name(name: str) -> tuple[str, int]:
    """'00124_60.png' -> ('00124', 60). Prefix is ASSUMED to be the patient ID."""
    m = NAME_RE.match(name)
    if m is None:
        raise ValueError(f"unexpected filename: {name}")
    return m.group(1), int(m.group(2))


def split_dir(cfg: dict, key: str, kind: str = "img") -> Path:
    sub = cfg["splits"][key]["img_dir" if kind == "img" else "mask_dir"]
    if sub is None:
        raise KeyError(f"{key} has no {kind} dir")
    return Path(cfg["root"]) / key / sub


def list_images(cfg: dict, key: str) -> list[Path]:
    return sorted(split_dir(cfg, key, "img").glob("*.png"))


def mask_path(cfg: dict, key: str, img_path: Path) -> Path | None:
    if cfg["splits"][key]["mask_dir"] is None:
        return None
    return split_dir(cfg, key, "mask") / img_path.name


def mask_to_binary(mask: np.ndarray) -> np.ndarray:
    """RGBA/RGB/gray uint8 mask -> {0,1} uint8. Alpha ignored.

    Two of the three lesion colours are dark (max channel 77 and 143), so a
    '>127' threshold would silently drop labelled pixels. Use '!= 0'.
    """
    if mask.ndim == 3:
        mask = mask[..., :3].max(axis=-1)
    return (mask != 0).astype(np.uint8)


def load_mask_binary(path: str | Path) -> np.ndarray:
    return mask_to_binary(np.asarray(Image.open(path)))