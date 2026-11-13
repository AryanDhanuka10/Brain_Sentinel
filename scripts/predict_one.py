"""Analyse one PNG slice with a trained run, using the same agent graph as the API (no server needed).

    python scripts/predict_one.py --image slice.png [--run artifacts/models/<run>] [--out out_dir]
Writes report.json and original/heatmap/overlay PNGs to --out (default: artifacts/predict_one/<report_id>).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from brainsentinel.agents.graph import build_graph, run_analysis  # noqa: E402
from brainsentinel.inference import Predictor, resolve_model_dir  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--run")
    ap.add_argument("--out")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    run = Path(a.run) if a.run else resolve_model_dir(REPO)
    if run is None:
        print("No trained model: pass --run or create artifacts/selected_model.json (scripts/select_model.py).", file=sys.stderr)
        return 2
    import torch
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    pred = Predictor(run, dev)
    st = run_analysis(build_graph(pred, pred.palette), Path(a.image).read_bytes(), Path(a.image).name)
    rep = dict(st["report"], trace=st["trace"])
    out = Path(a.out or REPO / "artifacts" / "predict_one" / rep["report_id"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    for k in ("original", "heatmap", "overlay"):
        if st.get("images"):
            (out / f"{k}.png").write_bytes(st["images"][k])
    print(f"status: {st['status']}\n" + "\n".join("- " + s["text"] for s in rep["statements"]) + f"\nwritten to {out}")
    return 0 if st["status"] not in ("rejected_input", "inference_failed", "no_model_available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
