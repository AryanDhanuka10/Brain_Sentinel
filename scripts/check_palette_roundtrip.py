"""Day 2 step 1: palette bijection + exact RGB round-trip for every BMAD Brain image."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from brainsentinel.data.palette import Palette, load_rgb, pack_rgb  # noqa: E402

RAW = ROOT / "data" / "raw" / "Brain"
EXPECTED_TOTAL = 11298


def image_files() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(RAW.rglob("*.png")):
        rel = p.relative_to(RAW).parts
        if "label" in rel[:-1]:
            continue  # masks
        groups["/".join(rel[:2])].append(p)
    return groups


def main() -> int:
    groups = image_files()
    total = sum(len(v) for v in groups.values())
    print({g: len(v) for g, v in sorted(groups.items())}, "total", total)

    seen = np.zeros(1 << 24, dtype=bool)  # palette from TRAIN only
    for g, ps in groups.items():
        if g.startswith("train"):
            for p in ps:
                seen[pack_rgb(load_rgb(p)).ravel()] = True
    pal = Palette.from_keys(np.flatnonzero(seen))
    print("train colours:", len(pal.colors), "monotone:", pal.is_monotone())

    import matplotlib
    bone = np.rint(matplotlib.colormaps["bone"](np.arange(len(pal.colors)) / 255)[:, :3] * 255)
    bone_diff = int(np.abs(pal.colors.astype(int) - bone.astype(int)).max())

    bad: list[dict] = []
    per_group: dict[str, dict] = {}
    for g, ps in sorted(groups.items()):
        seen_g = np.zeros(1 << 24, dtype=bool)
        n_ok = 0
        for p in ps:
            rgb = load_rgb(p)
            seen_g[pack_rgb(rgb).ravel()] = True
            try:
                exact = np.array_equal(pal.decode(pal.encode(rgb)), rgb)
                reason = "decode != original"
            except ValueError as e:
                exact, reason = False, str(e)
            if exact:
                n_ok += 1
            else:
                bad.append({"file": str(p.relative_to(RAW)), "reason": reason})
        per_group[g] = {"n": len(ps), "exact": n_ok, "colours": int(seen_g.sum())}

    out = {
        "total_images": total, "train_palette_size": int(len(pal.colors)),
        "monotone": pal.is_monotone(), "max_abs_diff_vs_bone_rounded": bone_diff,
        "per_group": per_group, "failures": bad[:50], "n_failures": len(bad),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "day2_palette.json").write_text(json.dumps(out, indent=2))
    for g, d in per_group.items():
        print(g, d)
    print("max |palette - round(bone)|:", bone_diff)
    ok = (not bad) and total == EXPECTED_TOTAL and len(pal.colors) == 256
    print("PASS" if ok else f"FAIL (failures={len(bad)}, total={total}, K={len(pal.colors)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
