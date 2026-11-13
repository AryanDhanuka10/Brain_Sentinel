"""Train one anomaly detector on the NORMAL training split (train/good) of the local BMAD Brain data.

    python scripts/train.py --model patchcore
    python scripts/train.py --model padim --tag pca99 --set hparams.reduction=pca hparams.pca_variance=0.99
Writes artifacts/models/<run_id>/ (model.pt, config.json, ...) and appends results/experiment_log.csv.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from _common import apply_overrides, load_model_cfg  # noqa: E402
from brainsentinel.pipeline import (EXPERIMENT_COLS, Context, append_row, run_id_for, train_run)  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["autoencoder", "padim", "patchcore"])
    ap.add_argument("--config", help="default configs/<model>.yaml")
    ap.add_argument("--tag", default="", help="run-name suffix, e.g. pca99")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(REPO / "artifacts" / "models"))
    ap.add_argument("--set", nargs="*", help="dotted overrides, e.g. hparams.reduction=pca")
    ap.add_argument("--data-config"); ap.add_argument("--pp-config"); ap.add_argument("--palette-cache")
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    a = ap.parse_args(argv)
    results = Path(a.results_dir)

    cpath = Path(a.config or REPO / "configs" / f"{a.model}.yaml")
    mc = apply_overrides(load_model_cfg(cpath), a.set)
    seed = a.seed if a.seed is not None else int(mc["train"].get("seed", 0))
    if "seed" in mc["hparams"]:          # only models that take a seed hyper-parameter (the AE does not)
        mc["hparams"]["seed"] = seed
    ctx = Context.from_files(a.data_config, a.pp_config, a.palette_cache)
    if mc.get("normalize_mode"):
        ctx.pp = copy.deepcopy(ctx.pp)
        ctx.pp["normalize"]["mode"] = mc["normalize_mode"]
    rid = run_id_for(a.model, a.tag, seed)
    status, notes, cj = "OK", "", {}
    try:
        cj = train_run(ctx, a.model, mc["hparams"], mc.get("fit", {}), Path(a.out) / rid, seed=seed, device=a.device,
                       batch_size=int(mc["train"].get("batch_size", 16)), tag_config_path=str(cpath.relative_to(REPO)
                                                                                           if cpath.is_relative_to(REPO) else cpath))
        print(f"trained {rid}: {cj['runtime_s']:.1f}s, checkpoint {cj['checkpoint_mb']:.1f} MB, peak VRAM {cj['peak_vram_gb']}")
    except BaseException as e:  # noqa: BLE001 - incl. Ctrl-C: an interrupted run must never be logged OK
        status, notes = ("INTERRUPTED" if isinstance(e, KeyboardInterrupt) else "FAILED"), f"{type(e).__name__}: {e}"
        raise
    finally:
        append_row(results / "experiment_log.csv", EXPERIMENT_COLS, {
            "run_id": rid, "date": time.strftime("%Y-%m-%d"), "model": a.model, "config_path": str(cpath),
            "seed": seed, "checkpoint": str(Path(a.out) / rid / "model.pt"), "metrics_file": "",
            "runtime_s": cj.get("runtime_s", ""), "peak_vram_gb": cj.get("peak_vram_gb", ""),
            "status": status, "failure_notes": notes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
