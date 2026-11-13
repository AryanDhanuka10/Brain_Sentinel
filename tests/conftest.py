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


# ---- shared NON-MEDICAL mini-dataset fixtures (code-path tests only; see tests/fixtures_minidataset.py) ----
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def mini_ctx(tmp_path_factory):
    from brainsentinel.data.preprocessing import load_pp_config
    from brainsentinel.pipeline import Context
    from tests import fixtures_minidataset as fx
    root = tmp_path_factory.mktemp("mini_session")
    cfg, pal = fx.build(root)
    pp = load_pp_config()
    pp["index_csv"] = str(root / "index.csv")
    c = Context(cfg, pp, pal, root / "index.csv")
    c.root = root
    return c


@pytest.fixture(scope="session")
def mini_run(mini_ctx):
    """A PatchCore run (random-init backbone: plumbing only) trained, with validation thresholds frozen."""
    from brainsentinel.pipeline import evaluate_run, train_run
    d = mini_ctx.root / "models" / "patchcore_mini_s0"
    train_run(mini_ctx, "patchcore", {"pretrained": False, "coreset_ratio": 0.05}, {"max_images": 12}, d, seed=0,
              device="cpu", batch_size=8, log=lambda *_: None)
    evaluate_run(mini_ctx, d, "valid", device="cpu", n_bins=512, n_boot=20, bench=False)
    return d


@pytest.fixture(scope="session")
def mini_png(mini_ctx):
    """name -> PNG bytes of real files from the mini dataset."""
    root = mini_ctx.root / "Brain"
    return {"normal": (root / "test/good/img/00400_60.png").read_bytes(),
            "lesion": (root / "test/Ungood/img/00500_60.png").read_bytes()}
