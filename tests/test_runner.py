"""Runner/protocol tests on NON-MEDICAL fixtures: collection, evaluation variants, test-split guard."""
import json

import numpy as np
import pytest
import torch

from brainsentinel.evaluation.runner import (benchmark_latency, collect_predictions, evaluate_predictions,
                                             guard_test_run, load_predictions, log_evaluation, select_and_freeze)
from brainsentinel.models import build_model
from tests.fixtures_nonmedical import batches, field

S = 64


def _mk_batches(n_norm, n_abn, seed, patients_offset=0):
    """Dict batches shaped like BMADBrain items. Patients: 2 slices each."""
    xs, ms, rows = [], [], []
    for i in range(n_norm + n_abn):
        abn = i >= n_norm
        x = field(1, S, seed * 100 + i, square=(10 + i % 5, 12, 10) if abn else None)
        m = torch.zeros(1, S, S, dtype=torch.uint8)
        if abn:
            m[0, 10 + i % 5:20 + i % 5, 12:22] = 1
        xs.append(x), ms.append(m)
        rows.append(dict(file=f"{i:05d}_{60 + 8 * (i % 2)}.png", key="valid/Ungood" if abn else "valid/good",
                         patient=f"{patients_offset + i // 2:05d}", slice=60 + 8 * (i % 2),
                         slice_bin="60s" if i % 2 == 0 else "70s", corner_bg=0))
    out = []
    for a in range(0, len(xs), 4):
        sl = slice(a, a + 4)
        b = {"image": torch.cat(xs[sl]), "mask": torch.cat(ms[sl]), "brain": torch.ones(len(xs[sl]), S, S, dtype=torch.bool)}
        for k in ("file", "key", "patient", "slice", "slice_bin", "corner_bg"):
            b[k] = [r[k] for r in rows[sl]]
        out.append(b)
    return out, len(xs)


@pytest.fixture(scope="module")
def fitted():
    m = build_model("patchcore", {"pretrained": False, "input_size": S, "coreset_ratio": 0.2})
    m.fit(batches(field(20, S, 0)), device="cpu")
    return m


def test_end_to_end_valid_then_test(fitted, tmp_path):
    vb, vn = _mk_batches(8, 8, 1)
    vdir = collect_predictions(fitted, vb, vn, tmp_path / "pred_valid", size=S)
    thr = select_and_freeze(vdir, tmp_path / "threshold.json", n_bins=512)
    assert thr["selected_on"] == "valid"
    tb, tn = _mk_batches(12, 12, 2, patients_offset=50)
    tdir = collect_predictions(fitted, tb, tn, tmp_path / "pred_test", size=S)
    res = evaluate_predictions(tdir, guard_test_run(tmp_path), n_bins=512, n_boot=50, exclude_patients=["00050", "00051"])
    off = res["official"]
    assert off["image"]["image_auroc"] > 0.9                    # trivial fixture: sanity, not a result
    assert {"pixel_auroc", "pixel_aupr", "aupro_fpr0.3", "dice_pooled", "dice_per_image"} <= set(off["pixel_full_frame"])
    assert "pixel_brain_roi" in off and "img_sensitivity" in off["image"] and "img_fpr" in off["image"]
    assert res["patient_disjoint"]["n_excluded"] == 4 and res["patient_disjoint"]["n_images"] == 20
    assert set(res["slice_stratified_image"]) == {"60s", "70s"}


def test_predictions_align_with_masks_and_meta(fitted, tmp_path):
    vb, vn = _mk_batches(4, 4, 3)
    P = load_predictions(collect_predictions(fitted, vb, vn, tmp_path / "p", size=S))
    assert P["maps"].shape == P["masks"].shape == (8, S, S) and len(P["meta"]) == 8
    assert P["labels"].tolist() == [0] * 4 + [1] * 4 and int(P["masks"][:4].sum()) == 0


def test_count_mismatch_is_an_error(fitted, tmp_path):
    vb, vn = _mk_batches(4, 4, 3)
    with pytest.raises(ValueError):
        collect_predictions(fitted, vb, vn + 1, tmp_path / "p", size=S)


def test_test_guard_requires_frozen_valid_thresholds(tmp_path):
    with pytest.raises(PermissionError):
        guard_test_run(tmp_path)
    (tmp_path / "threshold.json").write_text(json.dumps({"selected_on": "test", "image_threshold": 1, "pixel_threshold": 1}))
    with pytest.raises(ValueError):
        guard_test_run(tmp_path)


def test_test_guard_blocks_second_test_run_unless_allowed(tmp_path):
    (tmp_path / "threshold.json").write_text(json.dumps({"selected_on": "valid", "image_threshold": 1, "pixel_threshold": 1}))
    guard_test_run(tmp_path)
    log_evaluation(tmp_path, "valid")
    guard_test_run(tmp_path)                                     # a valid run does not consume the test run
    log_evaluation(tmp_path, "test")
    with pytest.raises(PermissionError):
        guard_test_run(tmp_path)
    guard_test_run(tmp_path, allow_rerun=True)


def test_latency_benchmark_reports_positive_numbers(fitted):
    r = benchmark_latency(fitted, field(1, S, 5), "cpu", warmup=1, runs=3)
    assert r["latency_ms_mean"] > 0 and r["latency_ms_p95"] >= r["latency_ms_mean"] * 0.5 and r["peak_vram_gb"] is None
