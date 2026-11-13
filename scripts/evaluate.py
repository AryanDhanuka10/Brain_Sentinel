"""Evaluate a trained run on the official validation or test split.

    python scripts/evaluate.py --run artifacts/models/patchcore_s0 --split valid   # selects + freezes thresholds
    python scripts/evaluate.py --run artifacts/models/patchcore_s0 --split test    # ONLY after model selection
The test split needs the frozen validation thresholds and is blocked after its first evaluation
(--allow-rerun overrides and is logged).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from brainsentinel.pipeline import (METRICS_COLS, Context, append_row, evaluate_run, metrics_row)  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True, choices=["valid", "test"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--allow-rerun", action="store_true")
    ap.add_argument("--no-bench", action="store_true", help="skip latency/VRAM benchmark")
    ap.add_argument("--data-config"); ap.add_argument("--pp-config"); ap.add_argument("--palette-cache")
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    a = ap.parse_args(argv)
    results = Path(a.results_dir)

    ctx = Context.from_files(a.data_config, a.pp_config, a.palette_cache)
    cj = json.loads((Path(a.run) / "config.json").read_text())
    ctx.pp["normalize"]["mode"] = cj["normalize"]["mode"]            # evaluate with the run's own normalisation
    res = evaluate_run(ctx, a.run, a.split, device=a.device, batch_size=a.batch_size, n_boot=a.n_boot,
                       allow_rerun=a.allow_rerun, bench=not a.no_bench)
    append_row(results / "metrics.csv", METRICS_COLS, metrics_row(res))
    o = res["official"]
    print(json.dumps({"run": res["run_id"], "split": a.split, "image": {k: o["image"][k] for k in ("image_auroc", "image_aupr")},
                      "pixel": {k: o["pixel_full_frame"][k] for k in ("pixel_auroc", "pixel_aupr", "aupro_fpr0.3", "dice_per_image", "iou_per_image")},
                      "engineering": res.get("engineering")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
