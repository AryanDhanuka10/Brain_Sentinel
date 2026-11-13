"""Derive the 256-colour palette from train/good and cache it at artifacts/palette.json (used by inference)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from brainsentinel.data.checks import load_config  # noqa: E402
from brainsentinel.data.preprocessing import resolve_cfg  # noqa: E402
from brainsentinel.pipeline import export_palette  # noqa: E402

if __name__ == "__main__":
    pal = export_palette(resolve_cfg(load_config(REPO / "configs" / "data.yaml")), REPO / "artifacts" / "palette.json")
    print("wrote artifacts/palette.json,", len(pal.colors), "colours")
