"""Per-image metadata index for BMAD Brain (rebuildable; output is gitignored)."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


from .checks import list_images, mask_path, mask_to_binary, parse_name
from .palette import Palette, corner_bg, load_rgb, pack_rgb

BASE_COLUMNS = ["split", "class", "file", "patient", "slice", "slice_bin",
                "lesion_px", "size_bin"]
TAIL_COLUMNS = ["unknown_mask_px", "mask_missing", "dims_ok", "sha256",
                "overlap_patient", "dup_cross_split", "dup_within_split", "corner_bg"]


def fg_colors(cfg: dict) -> list[tuple[int, int, int]]:
    return [tuple(c) for c in cfg["mask"]["allowed_rgb"] if tuple(c) != (0, 0, 0)]


def fg_col(rgb) -> str:
    return "fg_%d_%d_%d" % tuple(rgb)


def columns(cfg: dict) -> list[str]:
    return BASE_COLUMNS + [fg_col(c) for c in fg_colors(cfg)] + TAIL_COLUMNS


def slice_bin(s: int) -> str:
    return f"{s // 10 * 10}s"


def size_bin(px: int | None) -> str:
    """A-priori reporting bins (not tuned): sub-100 px matches the Day 1 caveat."""
    if px is None:
        return "na"
    if px == 0:
        return "0"
    if px < 100:
        return "1-99"
    if px < 1000:
        return "100-999"
    if px < 3000:
        return "1000-2999"
    return "3000+"


def _mask_info(path: Path, fg: list[tuple[int, int, int]]):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGBA"))
    keys = pack_rgb(arr)
    lesion = int(mask_to_binary(arr).sum())  # project rule: RGB != 0
    counts = {fg_col(c): int((keys == ((c[0] << 16) | (c[1] << 8) | c[2])).sum()) for c in fg}
    unknown = lesion - sum(counts.values())  # nonzero pixels with a non-allowed colour
    return lesion, counts, unknown, tuple(arr.shape[:2])


def build_rows(cfg: dict, pal: Palette) -> list[dict]:
    size = cfg["image_size"]
    fg = fg_colors(cfg)
    rows: list[dict] = []
    for key in cfg["splits"]:
        split, cls = key.split("/")
        for p in list_images(cfg, key):
            patient, sl = parse_name(p.name)
            with Image.open(p) as im:
                w, h = im.size
            row = {"split": split, "class": cls, "file": p.name, "patient": patient,
                   "slice": sl, "slice_bin": slice_bin(sl),
                   "lesion_px": None, "unknown_mask_px": None, "mask_missing": 0,
                   "dims_ok": int((h, w) == (size, size)),
                   "corner_bg": corner_bg(pal.encode(load_rgb(p))),
                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for c in fg:
                row[fg_col(c)] = None
            mp = mask_path(cfg, key, p)
            if mp is not None:
                if not mp.exists():
                    row["mask_missing"] = 1
                else:
                    lesion, counts, unknown, hw = _mask_info(mp, fg)
                    row.update(lesion_px=lesion, unknown_mask_px=unknown, **counts)
                    row["dims_ok"] = int(bool(row["dims_ok"]) and hw == (size, size))
            row["size_bin"] = size_bin(row["lesion_px"])
            rows.append(row)
    return rows


def add_flags(rows: list[dict]) -> set[str]:
    """Add duplicate and overlap flags in place. Returns the valid∩test patient set."""
    by_hash: dict[str, list[dict]] = {}
    for r in rows:
        by_hash.setdefault(r["sha256"], []).append(r)
    for grp in by_hash.values():
        n_splits = len({x["split"] for x in grp})
        for r in grp:
            same = sum(1 for x in grp if x["split"] == r["split"])
            r["dup_cross_split"] = int(n_splits > 1)
            r["dup_within_split"] = int(same > 1)
    pv = {r["patient"] for r in rows if r["split"] == "valid"}
    pt = {r["patient"] for r in rows if r["split"] == "test"}
    overlap = pv & pt
    for r in rows:
        r["overlap_patient"] = int(r["split"] in ("valid", "test") and r["patient"] in overlap)
    return overlap


def write_csv(rows: list[dict], cols: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r[c]) for c in cols})
