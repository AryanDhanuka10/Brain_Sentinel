import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


@pytest.fixture(scope="session")
def cfg():
    c = yaml.safe_load((REPO / "configs" / "data.yaml").read_text())
    c["root"] = str(REPO / c["root"])
    if not Path(c["root"]).is_dir():
        pytest.skip(f"dataset not found at {c['root']}")
    return c