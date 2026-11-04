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


@pytest.fixture(scope="session")
def train_pal(cfg):
    from brainsentinel.data.palette import train_palette
    return train_palette(cfg)


@pytest.fixture(scope="session")
def pp():
    from brainsentinel.data.preprocessing import load_pp_config
    return load_pp_config()


@pytest.fixture(scope="session")
def index_rows():
    import csv
    p = REPO / "data" / "processed" / "index.csv"
    if not p.exists():
        pytest.skip("run scripts/build_index.py first")
    with p.open(newline="") as f:
        return list(csv.DictReader(f))
