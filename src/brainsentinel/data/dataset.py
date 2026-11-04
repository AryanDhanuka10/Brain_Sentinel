"""BMAD Brain dataset: deterministic, no augmentation, guarded against split leakage."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .checks import list_images, load_mask_binary, mask_path, split_dir
from .palette import Palette
from .preprocessing import REPO, get_norm, load_index_image, resolve_cfg, to_network_input


def read_index(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def assert_split_isolation(rows: Sequence[dict]) -> None:
    """Train patients must be disjoint from valid and test.

    The documented valid/test overlap (10 patients) is NOT an error here: splits stay official and the
    overlap is handled by the sensitivity run (exclude_patients).
    """
    by: dict[str, set[str]] = {"train": set(), "valid": set(), "test": set()}
    for r in rows:
        by[r["split"]].add(r["patient"])
    bad = by["train"] & (by["valid"] | by["test"])
    if bad:
        raise ValueError(f"train patients also in valid/test: {sorted(bad)[:5]} ({len(bad)} total)")


class BMADBrain(Dataset):
    """Items come ONLY from index.csv, and each split's index rows must equal its files on disk.

    keys: e.g. ["train/good"] (images only) or ["valid/good", "valid/Ungood"] (image + binary mask).
    exclude_patients: dropped AFTER the guards (used for the patient-disjoint sensitivity run).
    """

    def __init__(self, cfg: dict, pp: dict, pal: Palette, keys: Iterable[str],
                 index_csv: str | Path | None = None, exclude_patients: Iterable[str] = ()):
        self.cfg = resolve_cfg(cfg)
        self.pp = pp
        self.pal = pal
        self.size = int(pp["input"]["size"])
        self.keys = list(keys)
        has_mask = {self.cfg["splits"][k]["mask_dir"] is not None for k in self.keys}
        if len(has_mask) != 1:
            raise ValueError("cannot mix keys with and without masks")
        self.with_mask = has_mask.pop()
        self.mean, self.std = get_norm(pp)

        rows = read_index(Path(index_csv) if index_csv else REPO / pp["index_csv"])
        assert_split_isolation(rows)
        excl = set(exclude_patients)
        items: list[tuple[str, dict]] = []
        for k in self.keys:
            krows = [r for r in rows if f"{r['split']}/{r['class']}" == k]
            on_disk = {p.name for p in list_images(self.cfg, k)}
            in_index = {r["file"] for r in krows}
            if on_disk != in_index or len(krows) != len(in_index):
                raise ValueError(f"{k}: disk/index mismatch (disk-only {len(on_disk - in_index)}, "
                                 f"index-only {len(in_index - on_disk)}, rows {len(krows)})")
            if len(krows) != self.cfg["splits"][k]["expected_n"]:
                raise ValueError(f"{k}: {len(krows)} rows, expected {self.cfg['splits'][k]['expected_n']}")
            items += [(k, r) for r in sorted(krows, key=lambda r: r["file"]) if r["patient"] not in excl]
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        k, r = self.items[i]
        path = split_dir(self.cfg, k, "img") / r["file"]
        idx, brain, bg = load_index_image(path, self.pal, self.size)
        if bg != int(r["corner_bg"]):
            raise ValueError(f"{path.name}: corner_bg {bg} != index.csv {r['corner_bg']}")
        out = {
            "image": torch.from_numpy(to_network_input(idx, self.mean, self.std)),  # (3,H,W) float32
            "brain": torch.from_numpy(brain),                                       # (H,W) bool, metadata
            "patient": r["patient"], "slice": int(r["slice"]), "slice_bin": r["slice_bin"],
            "corner_bg": bg, "key": k, "file": r["file"],
        }
        if self.with_mask:
            m = load_mask_binary(mask_path(self.cfg, k, path))                      # (H,W) uint8 {0,1}
            if m.shape != idx.shape:
                raise ValueError(f"{path.name}: mask {m.shape} != image {idx.shape}")
            out["mask"] = torch.from_numpy(m)
        return out


def make_loader(ds: Dataset, batch_size: int, shuffle: bool = False, seed: int = 0,
                num_workers: int = 0) -> DataLoader:
    """Seeded loader: same seed -> same order. num_workers=0 by default (reads are ~1 ms)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())
