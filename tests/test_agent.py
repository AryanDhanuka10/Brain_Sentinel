"""LangGraph workflow: every branch, grounding, LLM guards. Uses the NON-MEDICAL mini dataset (plumbing only)."""
import io
import re
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from brainsentinel.agents.evidence_agent import route_decision
from brainsentinel.agents.graph import build_graph, run_analysis
from brainsentinel.agents.narrate import safe_narrative
from brainsentinel.agents.report_agent import _get
from brainsentinel.inference import Predictor
from brainsentinel.inference.checks import InputRejected, assess_quality, decode_upload
from brainsentinel.inference import evidence as ev

BANNED = re.compile(r"tumou?r|cancer|malignan|diagnos(?!is\b)|patient has|probability of", re.I)


@pytest.fixture(scope="module")
def pred(mini_run):
    return Predictor(mini_run, "cpu")


def _png(arr, mode):
    b = io.BytesIO()
    Image.fromarray(arr, mode).save(b, format="PNG")
    return b.getvalue()


def _nodes(state):
    return [t["node"] for t in state["trace"]]


# ---------------------------------------------------------------- happy paths
def test_full_path_trace_report_and_human_review(pred, mini_png):
    st = run_analysis(build_graph(pred, pred.palette), mini_png["normal"], "slice.png")
    assert _nodes(st)[:8] == ["validate_input", "verify_domain", "assess_quality", "select_model", "run_detector",
                              "make_heatmap", "analyze_evidence", "assess_confidence"]
    assert _nodes(st)[-2:] == ["build_report", "human_review"]
    assert st["status"] in ("no_region_flagged", "flagged_for_review", "uncertain_escalated")
    r = st["report"]
    assert r["human_review"]["status"] == "pending" and r["inference"]["score"] == pytest.approx(float(st["inference"]["score"]))
    assert r["inference"]["pixel_threshold"] == pred.thresholds["pixel_threshold"]
    assert st["amap"].shape == (240, 240) and {"original", "heatmap", "overlay"} <= set(st["images"])
    for k in ("original", "heatmap", "overlay"):
        assert Image.open(io.BytesIO(st["images"][k])).size == (240, 240)
    assert st["stability"]["n_variants"] == 7


def test_statements_are_grounded_and_free_of_clinical_claims(pred, mini_png):
    for name in ("normal", "lesion"):
        r = run_analysis(build_graph(pred, pred.palette), mini_png[name])["report"]
        assert r["statements"] and "not a diagnosis" in r["disclaimer"]
        for s in r["statements"]:
            assert s["refs"], s
            for ref in s["refs"]:
                _get(r, ref)                              # resolves
            assert not BANNED.search(s["text"]), s["text"]
        txt = " ".join(s["text"] for s in r["statements"])
        assert f"{r['inference']['score']:.4f}" in txt     # numbers come from the measured output


def test_route_follows_measured_evidence(pred, mini_png):
    """Move the frozen threshold around the actual score: routing must follow, never contradict, the evidence."""
    base = run_analysis(build_graph(pred, pred.palette), mini_png["lesion"])
    s = base["inference"]["score"]
    import copy
    for factor in (0.2, 5.0):
        p2 = copy.copy(pred)
        p2.thresholds = dict(pred.thresholds, image_threshold=s * factor)
        st = run_analysis(build_graph(p2, p2.palette), mini_png["lesion"])
        e = st["evidence"]
        if factor == 5.0:                                   # threshold far above the score
            assert e["flagged"] is False and st["status"] in ("no_region_flagged", "uncertain_escalated")
        else:                                               # threshold far below
            assert e["flagged"] is True and st["status"] in ("flagged_for_review", "uncertain_escalated")
        assert (st["status"] == "uncertain_escalated") == (route_decision(st) == "uncertain")


@pytest.mark.parametrize("ev_in,expect", [
    ({"flagged": True, "strength": "strong", "reasons": []}, "flagged"),
    ({"flagged": True, "strength": "moderate", "reasons": ["x"]}, "flagged"),
    ({"flagged": True, "strength": "weak", "reasons": ["unstable"]}, "uncertain"),
    ({"flagged": False, "strength": "not_flagged", "reasons": []}, "clear"),
    ({"flagged": False, "strength": "not_flagged", "reasons": ["unstable"]}, "uncertain"),
    ({"flagged": None, "strength": "unavailable", "reasons": ["no threshold"]}, "uncertain"),
])
def test_route_table(ev_in, expect):
    assert route_decision({"evidence": ev_in}) == expect


def test_evidence_rules_unit():
    good = {"mean_map_corr": 0.99, "min_map_corr": 0.97, "score_cv": 0.01, "max_peak_shift_px": 1.0}
    loc = {"outside_brain_frac": 0.0, "border_frac": 0.0}
    assert ev.assess_evidence(2.0, 1.0, good, loc)["strength"] == "strong"
    assert ev.assess_evidence(1.1, 1.0, good, loc)["strength"] == "moderate"
    assert ev.assess_evidence(0.5, 1.0, good, loc)["strength"] == "not_flagged"
    assert ev.assess_evidence(2.0, 1.0, dict(good, mean_map_corr=0.5), loc)["strength"] == "weak"
    assert ev.assess_evidence(2.0, 1.0, good, dict(loc, border_frac=0.9))["strength"] == "weak"
    assert ev.assess_evidence(2.0, None, good, loc)["strength"] == "unavailable"


def test_perturbations_are_deterministic_and_conservative():
    idx = (np.arange(240 * 240) % 200).reshape(240, 240).astype(np.uint8)
    a, b = ev.perturbations(idx), ev.perturbations(idx)
    assert [x[0] for x in a] == [x[0] for x in b] and all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    assert all(abs(int(x[1].astype(int).mean()) - int(idx.astype(int).mean())) < 12 for x in a)


# ---------------------------------------------------------------- rejection / failure branches
@pytest.mark.parametrize("name,data,code", [
    ("empty", b"", "empty_file"),
    ("not_png", b"GIF89a....", "not_png"),
    ("truncated", None, "corrupt_image"),
    ("rgb_photo_like", "rgb", "wrong_size_or_encoding"),
    ("grayscale_240", "gray", "unsupported_encoding"),
    ("random_colours", "rand", "unsupported_encoding"),
])
def test_invalid_uploads_are_rejected_before_any_model_run(pred, mini_png, name, data, code):
    if name == "truncated":
        data = mini_png["normal"][:200]
    elif data == "rgb":
        data = _png(np.random.default_rng(0).integers(0, 255, (512, 384, 3), dtype=np.uint8), "RGB")
    elif data == "gray":
        data = _png(np.random.default_rng(0).integers(0, 255, (240, 240), dtype=np.uint8), "L")
    elif data == "rand":
        a = np.random.default_rng(0).integers(0, 255, (240, 240, 4), dtype=np.uint8)
        a[..., 3] = 255
        data = _png(a, "RGBA")
    with mock.patch.object(pred, "predict", wraps=pred.predict) as spy:
        st = run_analysis(build_graph(pred, pred.palette), data)
    assert st["status"] == "rejected_input" and spy.call_count == 0
    assert st["errors"][0]["code"] in ({code} if code != "wrong_size_or_encoding" else {"wrong_size"})
    assert st["report"]["human_review"]["status"] == "not_applicable" and st["report"]["inference"] is None
    assert _nodes(st) == ["validate_input", "build_report", "human_review"]


def test_quality_failure_stops_before_the_model(pred, mini_ctx):
    idx = np.full((240, 240), 0, np.uint8)
    idx[60:180, 60:180] = 100                                 # flat disc: no texture
    rgb = pred.palette.decode(idx)
    a = np.concatenate([rgb, np.full((240, 240, 1), 255, np.uint8)], 2)
    with mock.patch.object(pred, "predict", wraps=pred.predict) as spy:
        st = run_analysis(build_graph(pred, pred.palette), _png(a, "RGBA"))
    assert st["status"] == "quality_failed" and spy.call_count == 0 and st["report"]["human_review"]["status"] == "pending"
    assert "run_detector" not in _nodes(st) and st["quality"]["level"] == "fail"


def test_no_model_available_is_reported_not_faked(pred, mini_png):
    st = run_analysis(build_graph(None, pred.palette), mini_png["normal"])
    assert st["status"] == "no_model_available" and st["report"]["inference"] is None
    assert not st.get("images")


def test_inference_failure_retries_once_then_degrades(pred, mini_png):
    class Boom(Predictor):
        def __init__(self, base):
            self.__dict__.update(base.__dict__)
            self.n = 0

        def predict(self, idxs):
            self.n += 1
            raise RuntimeError("CUDA out of memory (simulated)")
    b = Boom(pred)
    st = run_analysis(build_graph(b, b.palette), mini_png["normal"])
    assert b.n == 2 and st["status"] == "inference_failed"
    assert [t["status"] for t in st["trace"] if t["node"] == "run_detector"] == ["retry", "fail"]
    assert st["report"]["inference"] is None and st["report"]["human_review"]["status"] == "pending"


def test_stability_failure_degrades_to_uncertain(pred, mini_png):
    class Flaky(Predictor):
        def __init__(self, base):
            self.__dict__.update(base.__dict__)

        def predict(self, idxs):
            if len(idxs) > 1:
                raise RuntimeError("batch too large (simulated)")
            return Predictor.predict(self, idxs)
    f = Flaky(pred)
    st = run_analysis(build_graph(f, f.palette), mini_png["normal"])
    assert st["stability"] is None and st["status"] == "uncertain_escalated"
    assert st["evidence"]["strength"] == "unavailable"


# ---------------------------------------------------------------- optional LLM
def test_llm_failure_never_breaks_the_result(pred, mini_png):
    def boom(_):
        raise TimeoutError("provider down")
    r = run_analysis(build_graph(pred, pred.palette, narrator=boom), mini_png["normal"])["report"]
    assert r["llm_narrative"]["accepted"] is False and r["inference"]["score"] > 0


@pytest.mark.parametrize("text,ok", [
    ("Tumor detected with 97% chance.", False), ("The score was 123456.789.", False), ("", False),
    ("x" * 2000, False), ("This is a diagnosis.", False)])
def test_narrative_guards_reject_unsafe_text(text, ok):
    assert (safe_narrative(text, "Image anomaly score 0.1234")[0] is not None) is ok


def test_narrative_accepts_grounded_text(pred, mini_png):
    def good(summary):
        return "The model produced an image anomaly score for this slice and the results are listed above for review."
    r = run_analysis(build_graph(pred, pred.palette, narrator=good), mini_png["normal"])["report"]
    assert r["llm_narrative"]["accepted"] is True and "language-model" in r["llm_narrative"]["label"]


# ---------------------------------------------------------------- checks module
def test_decode_roundtrip_matches_training_loader(pred, mini_ctx, mini_png):
    from brainsentinel.data.preprocessing import load_index_image
    p = mini_ctx.root / "Brain/test/good/img/00400_60.png"
    idx, brain, bg = load_index_image(p, pred.palette)
    d = decode_upload(p.read_bytes(), pred.palette)
    assert np.array_equal(d.idx, idx) and np.array_equal(d.brain, brain) and d.bg == bg


def test_quality_levels(pred, mini_png):
    q = assess_quality(decode_upload(mini_png["normal"], pred.palette))
    assert q["level"] in ("pass", "warn") and 0.1 < q["metrics"]["brain_fraction"] < 0.6


def test_predict_one_cli(mini_run, mini_ctx, tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import predict_one
    img = mini_ctx.root / "Brain/test/Ungood/img/00500_60.png"
    assert predict_one.main(["--image", str(img), "--run", str(mini_run), "--out", str(tmp_path / "o"), "--device", "cpu"]) == 0
    assert {"report.json", "original.png", "heatmap.png", "overlay.png"} <= {p.name for p in (tmp_path / "o").iterdir()}
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    assert predict_one.main(["--image", str(bad), "--run", str(mini_run), "--out", str(tmp_path / "o2"), "--device", "cpu"]) == 1


def test_report_has_plain_verdict_consistent_with_score_and_threshold(pred, mini_png):
    from brainsentinel.agents.report_agent import build_verdict
    for name in ("normal", "lesion"):
        r = run_analysis(build_graph(pred, pred.palette), mini_png[name])["report"]
        v, inf = r["verdict"], r["inference"]
        assert v["label"] == ("ANOMALY_DETECTED" if inf["score"] >= inf["image_threshold"] else "NO_ANOMALY_DETECTED")
        assert "not a diagnosis" in r["disclaimer"] and not BANNED.search(v["text"])
    base = {"outcome": {"status": "flagged_for_review"}, "inference": {"score": 2.27, "image_threshold": 2.24}}
    assert build_verdict(base)["borderline"] and build_verdict(base)["reliability"] == "low"
    far = {"outcome": {"status": "flagged_for_review"}, "inference": {"score": 4.0, "image_threshold": 2.24}}
    assert not build_verdict(far)["borderline"] and build_verdict(far)["reliability"] == "normal"
    assert build_verdict({"outcome": {"status": "rejected_input"}, "inference": None})["label"] == "INCONCLUSIVE"
