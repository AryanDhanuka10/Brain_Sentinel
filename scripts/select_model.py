"""Pick the final model from VALIDATION metrics only (predeclared rule in brainsentinel.pipeline.select_model).

    python scripts/select_model.py --runs artifacts/models/autoencoder_s0 artifacts/models/padim_s0 artifacts/models/patchcore_s0
Writes artifacts/selected_model.json and results/model_selection.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from brainsentinel.pipeline import select_model  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args(argv)
    sel = select_model(a.runs)
    (REPO / "artifacts").mkdir(exist_ok=True)
    (REPO / "artifacts" / "selected_model.json").write_text(json.dumps(sel, indent=2, default=str) + "\n")
    (REPO / "results" / "model_selection.json").write_text(json.dumps(sel, indent=2, default=str) + "\n")
    print("selected:", sel["selected"], "| rule:", json.dumps(sel["rule"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
