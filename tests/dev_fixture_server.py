"""DEV ONLY: serve the real API on a tiny NON-MEDICAL fixture model so the UI can be developed/tested without
your data. The model id is 'patchcore_mini_s0' and its images are discs, never MRI. Not for demos or results.

    python tests/dev_fixture_server.py [--port 8000]      # prints sample PNG paths to upload
"""
import argparse
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
import uvicorn  # noqa: E402

from brainsentinel.api.main import create_app  # noqa: E402
from brainsentinel.data.preprocessing import load_pp_config  # noqa: E402
from brainsentinel.inference import Predictor  # noqa: E402
from brainsentinel.pipeline import Context, evaluate_run, train_run  # noqa: E402
from tests import fixtures_minidataset as fx  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    root = Path(tempfile.mkdtemp(prefix="bs_fixture_"))
    cfg, pal = fx.build(root)
    pp = load_pp_config()
    pp["index_csv"] = str(root / "index.csv")
    ctx = Context(cfg, pp, pal, root / "index.csv")
    run = root / "models" / "patchcore_mini_s0"
    train_run(ctx, "patchcore", {"pretrained": False, "coreset_ratio": 0.05}, {"max_images": 12}, run, device="cpu",
              batch_size=8, log=lambda *_: None)
    evaluate_run(ctx, run, "valid", device="cpu", n_bins=512, n_boot=20, bench=False)
    print("FIXTURE MODEL (non-medical). Sample uploads:")
    for p in sorted((root / "Brain" / "test").glob("*/img/*_60.png"))[:4]:
        print("  ", p)
    uvicorn.run(create_app(predictor=Predictor(run, "cpu"), db_path=root / "api.db", storage_dir=root / "reports"),
                host="127.0.0.1", port=a.port, log_level="warning")
