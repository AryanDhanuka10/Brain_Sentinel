"""End-to-end protocol test on the NON-MEDICAL mini dataset: train x3 -> valid (freeze) -> test -> select.
Checks plumbing and guard rails, not model quality; the printed numbers are never reported anywhere."""
import json

import pytest

from brainsentinel.data.preprocessing import load_pp_config
from brainsentinel.pipeline import (Context, evaluate_run, metrics_row, select_model, train_run, append_row,
                                    METRICS_COLS, load_palette, export_palette)
from tests import fixtures_minidataset as fx


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    root = tmp_path_factory.mktemp("mini")
    cfg, pal = fx.build(root)
    pp = load_pp_config()
    pp["index_csv"] = str(root / "index.csv")
    c = Context(cfg, pp, pal, root / "index.csv")
    c.root = root
    return c


@pytest.fixture(scope="module")
def runs(ctx):
    base = ctx.root / "models"
    spec = {"autoencoder": ({"width": 8}, {"epochs": 2, "patience": 2}),
            "padim": ({"pretrained": False, "d_out": 20}, {}),
            "patchcore": ({"pretrained": False, "coreset_ratio": 0.05}, {"max_images": 12})}
    out = {}
    for kind, (hp, fk) in spec.items():
        out[kind] = base / kind
        train_run(ctx, kind, hp, fk, out[kind], seed=0, device="cpu", batch_size=8, log=lambda *_: None)
    return out


def test_train_writes_complete_run_dir(runs):
    for kind, d in runs.items():
        assert {"model.pt", "model.json", "config.json", "palette.json"} <= {p.name for p in d.iterdir()}
        cj = json.loads((d / "config.json").read_text())
        assert cj["trained_on"] == "train/good" and cj["normalize"]["mode"] and cj["runtime_s"] > 0
    assert (runs["autoencoder"] / "train_log.csv").exists()


def test_valid_freezes_thresholds_then_test_runs_once(ctx, runs):
    d = runs["patchcore"]
    with pytest.raises(PermissionError):
        evaluate_run(ctx, d, "test", device="cpu", bench=False)           # before thresholds exist
    rv = evaluate_run(ctx, d, "valid", device="cpu", n_bins=512, n_boot=30, bench=True)
    assert json.loads((d / "threshold.json").read_text())["selected_on"] == "valid"
    assert rv["official"]["image"]["image_auroc"] >= 0 and "latency_ms_mean" in rv["engineering"]
    rt = evaluate_run(ctx, d, "test", device="cpu", n_bins=512, n_boot=30, bench=False)
    assert rt["split"] == "test" and "pixel_brain_roi" in rt["official"]
    with pytest.raises(PermissionError):                                   # second test run is blocked
        evaluate_run(ctx, d, "test", device="cpu", bench=False)
    with pytest.raises(PermissionError):                                   # thresholds cannot be re-frozen after test
        evaluate_run(ctx, d, "valid", device="cpu", bench=False)
    evaluate_run(ctx, d, "test", device="cpu", n_bins=512, n_boot=30, bench=False, allow_rerun=True)


def test_selection_uses_only_valid_metrics_and_logs_rule(ctx, runs, tmp_path):
    for kind in ("autoencoder", "padim"):
        evaluate_run(ctx, runs[kind], "valid", device="cpu", n_bins=512, n_boot=30)
    sel = select_model([runs[k] for k in ("autoencoder", "padim", "patchcore")])
    assert sel["evaluated_on"] == "valid" and sel["selected"] in {c["run_id"] for c in sel["candidates"]}
    assert len(sel["candidates"]) == 3 and "aupro_margin" in sel["rule"]
    row = metrics_row(json.loads((runs["padim"] / "metrics_valid.json").read_text()))
    append_row(tmp_path / "metrics.csv", METRICS_COLS, row)
    assert (tmp_path / "metrics.csv").read_text().splitlines()[0].split(",") == METRICS_COLS


def test_selection_refuses_unevaluated_run(runs, tmp_path):
    with pytest.raises(FileNotFoundError):
        select_model([tmp_path])


def test_palette_cache_roundtrip_is_train_derived(ctx, tmp_path):
    p = tmp_path / "palette.json"
    export_palette(ctx.cfg, p)
    assert (load_palette(p).colors == ctx.pal.colors).all()
    d = json.loads(p.read_text())
    d["derived_from"] = "test/good"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_palette(p)


# ---- reporting / PCA rule / CLI plumbing -----------------------------------------------------------------
def test_comparison_markdown_marks_missing_as_not_run(ctx, runs):
    from brainsentinel.pipeline import comparison_markdown
    if not (runs["padim"] / "metrics_valid.json").exists():          # self-sufficient, order-independent
        evaluate_run(ctx, runs["padim"], "valid", device="cpu", n_bins=512, n_boot=30, bench=False)
    md = comparison_markdown([runs["padim"]])
    rows = [l for l in md.splitlines() if l.startswith("| ")]
    measured = [l for l in rows if l.startswith("| padim | padim |")]
    assert len(measured) == 1
    cells = [c.strip() for c in measured[0].strip("|").split("|")]
    assert all(c != "NOT RUN" for c in cells[2:9])                    # quality metrics were measured
    assert cells[9] == cells[10] == "NOT RUN" or float(cells[9]) > 0  # latency/VRAM only if benchmarked
    missing = [l for l in rows if l.startswith("| (none) |")]
    assert len(missing) == 2 + 3                                      # valid: AE+PatchCore; test: all three
    assert all(l.count("NOT RUN") == 9 for l in missing)
    assert "## valid" in md and "## test" in md and "not diagnosis" in md


def test_pca_decision_rule():
    from brainsentinel.pipeline import pca_decision
    base = dict(image_auroc=0.90, pixel_auroc=0.95, aupro=0.80, latency_ms=100.0, model_mb=100.0)
    ok = dict(image_auroc=0.895, pixel_auroc=0.945, aupro=0.795, latency_ms=70.0, model_mb=100.0)
    bad_quality = dict(ok, aupro=0.70)
    no_gain = dict(ok, latency_ms=95.0)
    assert pca_decision(base, ok)["keep"] is True
    assert pca_decision(base, bad_quality)["keep"] is False
    assert pca_decision(base, no_gain)["keep"] is False


def test_cli_train_and_evaluate_with_overrides(ctx, tmp_path):
    import sys
    from pathlib import Path
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import evaluate as ev
    import train as tr
    dcfg = tmp_path / "data.yaml"
    dcfg.write_text(yaml.safe_dump({**ctx.cfg, "root": ctx.cfg["root"]}))
    pp = dict(ctx.pp)
    pcfg = tmp_path / "pp.yaml"
    pcfg.write_text(yaml.safe_dump(pp))
    common = ["--data-config", str(dcfg), "--pp-config", str(pcfg), "--palette-cache", str(tmp_path / "pal.json"),
              "--results-dir", str(tmp_path / "results")]
    out = tmp_path / "models"
    tr.main(["--model", "padim", "--device", "cpu", "--out", str(out), "--tag", "cli", "--set", "hparams.pretrained=false",
             "hparams.d_out=16", *common])
    run = out / "padim_cli_s0"
    assert (run / "model.pt").exists()
    ev.main(["--run", str(run), "--split", "valid", "--device", "cpu", "--n-boot", "20", "--no-bench", *common])
    assert (run / "threshold.json").exists() and (tmp_path / "results" / "metrics.csv").exists()
    log = (tmp_path / "results" / "experiment_log.csv").read_text()
    assert "padim_cli_s0" in log and ",OK," in log


def test_run_carries_its_own_palette_even_if_repo_cache_differs(ctx, tmp_path, monkeypatch):
    """Regression: a stray artifacts/palette.json (e.g. the real one) must never leak into another context's run."""
    from brainsentinel import pipeline
    fake = tmp_path / "repo" / "artifacts"
    fake.mkdir(parents=True)
    (fake / "palette.json").write_text(json.dumps({"derived_from": "train/good",
                                                   "colors": [[i, i, i] for i in range(256)]}))
    monkeypatch.setattr(pipeline, "REPO", tmp_path / "repo")
    out = tmp_path / "run"
    train_run(ctx, "padim", {"pretrained": False, "d_out": 20}, {}, out, seed=0, device="cpu", batch_size=8,
              log=lambda *_: None)
    saved = json.loads((out / "palette.json").read_text())["colors"]
    assert saved == ctx.pal.colors.tolist()
