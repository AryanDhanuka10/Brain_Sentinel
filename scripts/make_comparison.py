"""Rebuild results/model_comparison.md from the metrics_<split>.json files of the given runs.

    python scripts/make_comparison.py --runs artifacts/models/*_s0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from brainsentinel.pipeline import comparison_markdown  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="*", default=[])
    a = ap.parse_args(argv)
    md = comparison_markdown(a.runs)
    (REPO / "results" / "model_comparison.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
