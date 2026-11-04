"""Day 3 full-corpus audit: EVERY image/mask of all 5 splits through the real BMADBrain path.
Run from repo root:  python scripts/audit_day3.py | tee results/day3_audit.log
Writes results/day3_audit.json. Exit code 1 if any hard check fails.
Descriptive only: test-split numbers must not drive any design choice.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from brainsentinel.data.checks import load_config  # noqa: E402
from brainsentinel.data.dataset import BMADBrain  # noqa: E402
from brainsentinel.data.palette import train_palette  # noqa: E402
from brainsentinel.data.preprocessing import get_norm, load_pp_config, resolve_cfg  # noqa: E402

KEYS = ["train/good", "valid/good", "valid/Ungood", "test/good", "test/Ungood"]
DAY2 = {"valid": (0, 93756), "test": (184, 6731065, 16)}  # lesion px on background / total (/ images)


def audit_key(cfg, pp, pal, key, mean, std) -> dict:
    ds = BMADBrain(cfg, pp, pal, [key])
    errs, bgs, frac = [], Counter(), []
    bad = dict(nonfinite=0, out_of_range=0, nonbinary=0, lesion_px_mismatch=0)
    les_total = les_bg = les_bg_imgs = 0
    t0 = time.time()
    for i in range(len(ds)):
        try:
            it = ds[i]
        except Exception as e:  # record, keep going
            errs.append((ds.items[i][1]["file"], repr(e)[:200]))
            continue
        x = it["image"].numpy()
        bad["nonfinite"] += int(not np.isfinite(x).all())
        x01 = x[0] * std[0] + mean[0]
        bad["out_of_range"] += int(x01.min() < -1e-5 or x01.max() > 1 + 1e-5)
        brain = it["brain"].numpy()
        bgs[it["corner_bg"]] += 1
        frac.append(float(brain.mean()))
        if "mask" in it:
            m = it["mask"].numpy()
            bad["nonbinary"] += int(not set(np.unique(m).tolist()) <= {0, 1})
            bad["lesion_px_mismatch"] += int(int(m.sum()) != int(ds.items[i][1]["lesion_px"]))
            n_bg = int(((m == 1) & ~brain).sum())
            les_total += int(m.sum())
            les_bg += n_bg
            les_bg_imgs += int(n_bg > 0)
    return {"n": len(ds), "errors": errs, **bad, "corner_bg_counts": {str(k): v for k, v in sorted(bgs.items())},
            "brain_fraction_mean": float(np.mean(frac)) if frac else None,
            "brain_fraction_std": float(np.std(frac)) if frac else None,
            "lesion_px_total": les_total, "lesion_px_on_background": les_bg,
            "images_with_lesion_on_background": les_bg_imgs, "seconds": round(time.time() - t0, 1)}


def main() -> int:
    cfg = resolve_cfg(load_config(REPO / "configs" / "data.yaml"))
    pp = load_pp_config()
    pal = train_palette(cfg)
    mean, std = get_norm(pp)
    res = {}
    for k in KEYS:
        res[k] = audit_key(cfg, pp, pal, k, mean, std)
        r = res[k]
        print(f"{k:13s} n={r['n']:5d} errors={len(r['errors'])} nonfinite={r['nonfinite']} "
              f"out_of_range={r['out_of_range']} nonbinary={r['nonbinary']} "
              f"lesion_px_mismatch={r['lesion_px_mismatch']} brain_frac={r['brain_fraction_mean']:.3f} "
              f"bg={r['corner_bg_counts']} ({r['seconds']}s)")
    hard = sum(len(r["errors"]) + r["nonfinite"] + r["out_of_range"] + r["nonbinary"] + r["lesion_px_mismatch"]
               for r in res.values())
    v_bg = sum(res[k]["lesion_px_on_background"] for k in KEYS[1:3])
    v_tot = sum(res[k]["lesion_px_total"] for k in KEYS[1:3])
    t_bg = sum(res[k]["lesion_px_on_background"] for k in KEYS[3:])
    t_tot = sum(res[k]["lesion_px_total"] for k in KEYS[3:])
    t_imgs = sum(res[k]["images_with_lesion_on_background"] for k in KEYS[3:])
    repro = {"valid": [v_bg, v_tot], "test": [t_bg, t_tot, t_imgs],
             "valid_matches_day2": (v_bg, v_tot) == DAY2["valid"],
             "test_matches_day2": (t_bg, t_tot, t_imgs) == DAY2["test"]}
    print(f"Day 2 reproduction: valid {v_bg}/{v_tot} (Day 2: {DAY2['valid'][0]}/{DAY2['valid'][1]}) "
          f"match={repro['valid_matches_day2']}; test {t_bg}/{t_tot} in {t_imgs} imgs "
          f"(Day 2: {DAY2['test'][0]}/{DAY2['test'][1]} in {DAY2['test'][2]}) match={repro['test_matches_day2']}")
    print("HARD CHECKS:", "ALL CLEAN" if hard == 0 else f"{hard} PROBLEMS (see json)")
    out = REPO / "results" / "day3_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"splits": res, "day2_reproduction": repro, "hard_problems": hard}, indent=2) + "\n")
    return 0 if hard == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
