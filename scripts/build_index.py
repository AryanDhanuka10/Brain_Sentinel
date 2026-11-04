"""Day 2 step 2: build data/processed/index.csv and run the data-contract checks."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from brainsentinel.data.palette import train_palette  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from brainsentinel.data.checks import load_config, split_dir  # noqa: E402
from brainsentinel.data.index import (add_flags, build_rows, columns, fg_col,  # noqa: E402
                                      fg_colors, slice_bin, write_csv)

DAY1_VALID_ABN_OVERLAP = 40  # Day 1 measured: 40 of 44 valid/Ungood images


def key_of(r: dict) -> str:
    return f"{r['split']}/{r['class']}"


def main() -> int:
    cfg = load_config()
    pal = train_palette(cfg)
    rows = build_rows(cfg, pal)
    overlap = add_flags(rows)
    write_csv(rows, columns(cfg), ROOT / "data" / "processed" / "index.csv")

    keys = list(cfg["splits"])
    ki = cfg["known_issues"]
    checks: list[tuple[str, bool, object]] = []

    def chk(name: str, ok: bool, detail: object = "") -> None:
        checks.append((name, bool(ok), detail))

    cnt = Counter(key_of(r) for r in rows)
    total_expected = sum(cfg["splits"][k]["expected_n"] for k in keys)
    chk("rows per split == expected_n", all(cnt[k] == cfg["splits"][k]["expected_n"] for k in keys), dict(cnt))
    chk("total rows", len(rows) == total_expected, f"{len(rows)} vs {total_expected}")

    orph = {}
    for k in keys:
        if cfg["splits"][k]["mask_dir"]:
            im = {p.name for p in split_dir(cfg, k, "img").glob("*.png")}
            mk = {p.name for p in split_dir(cfg, k, "mask").glob("*.png")}
            orph[k] = {"masks_without_image": len(mk - im), "images_without_mask": len(im - mk)}
    chk("image-mask pairing 1:1", all(v == 0 for d in orph.values() for v in d.values())
        and not any(r["mask_missing"] for r in rows), orph)
    chk("all dims == image_size", all(r["dims_ok"] for r in rows),
        f"bad={sum(1 for r in rows if not r['dims_ok'])}")
    unk = sum(r["unknown_mask_px"] or 0 for r in rows)
    chk("mask colours all in allowed_rgb", unk == 0, f"unknown px={unk}")
    tol = max(r["corner_bg"] for r in rows if r["split"] == "train")  # derived from TRAIN only
    chk("corner bg <= train-derived max", all(r["corner_bg"] <= tol for r in rows),
        {"train_max": tol, "n_offset": sum(1 for r in rows if r["corner_bg"] > 0)})

    good_nonempty = [r["file"] for r in rows if r["class"] == "good" and r["lesion_px"] not in (None, 0)]
    abn_blank = [r["file"] for r in rows if r["class"] == "Ungood" and not r["lesion_px"]]
    chk("good masks all empty", not good_nonempty, good_nonempty[:5])
    chk("no blank abnormal masks", not abn_blank, abn_blank[:5])

    ptr = {r["patient"] for r in rows if r["split"] == "train"}
    pvt = {r["patient"] for r in rows if r["split"] in ("valid", "test")}
    chk("train patients disjoint from valid/test", not (ptr & pvt), sorted(ptr & pvt)[:10])

    want_ids = set(cfg["patient_overlap"]["valid_test_ids"])
    chk("valid∩test patients == config ids", overlap == want_ids,
        {"extra": sorted(overlap - want_ids), "missing": sorted(want_ids - overlap)})
    ov = Counter(key_of(r) for r in rows if r["overlap_patient"])
    chk("valid/Ungood overlap rows == 40 (Day 1)", ov["valid/Ungood"] == DAY1_VALID_ABN_OVERLAP, dict(ov))
    chk("test overlap rows == config", sum(v for k, v in ov.items() if k.startswith("test"))
        == ki["sensitivity_exclusion_expected_test_images"], dict(ov))

    want_dup = set(ki["valid_test_duplicate_files"])
    dv = {r["file"] for r in rows if r["split"] == "valid" and r["dup_cross_split"]}
    n_dup = sum(r["dup_cross_split"] for r in rows)
    chk("cross-split duplicates == config list", dv == want_dup and n_dup == 2 * len(want_dup),
        {"files": len(dv), "rows": n_dup})
    chk("no train row in a cross-split duplicate",
        not any(r["dup_cross_split"] for r in rows if r["split"] == "train"))

    fg = fg_colors(cfg)
    summary: dict = {"rows": len(rows), "per_split": dict(cnt),
                     "within_split_duplicate_rows": sum(r["dup_within_split"] for r in rows),
                     "slice_bins": {k: dict(Counter(r["slice_bin"] for r in rows if key_of(r) == k)) for k in keys},
                     "lesion_px": {}, "fg_pixel_share": {}}
    for k in keys:
        if not cfg["splits"][k]["mask_dir"]:
            continue
        sub = [r for r in rows if key_of(r) == k]
        px = np.array([r["lesion_px"] for r in sub])
        if k.endswith("Ungood"):
            summary["lesion_px"][k] = {"n": len(px), "min": int(px.min()), "median": float(np.median(px)),
                                       "max": int(px.max()), "n_lt100": int((px < 100).sum())}
        tot = {fg_col(c): sum(r[fg_col(c)] for r in sub) for c in fg}
        s = sum(tot.values())
        summary["fg_pixel_share"][k] = {c: (round(v / s, 4) if s else None) for c, v in tot.items()}
    summary["checks"] = [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks]
    (ROOT / "results" / "day2_index_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    for n, ok, d in checks:
        print(("PASS" if ok else "FAIL"), n, "|", d)
    print("INFO within-split duplicate rows:", summary["within_split_duplicate_rows"])
    print("INFO lesion_px:", json.dumps(summary["lesion_px"]))
    print("INFO fg pixel share:", json.dumps(summary["fg_pixel_share"]))
    print("INFO slice bins:", json.dumps(summary["slice_bins"]))
    bad = [n for n, ok, _ in checks if not ok]
    print("RESULT:", "PASS" if not bad else f"FAIL {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
