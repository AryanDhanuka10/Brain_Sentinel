"""API tests (FastAPI TestClient). NON-MEDICAL mini-dataset model: plumbing only, no reported numbers."""
import io
import json
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from brainsentinel.api.main import MAX_BYTES, create_app
from brainsentinel.inference import Predictor
from brainsentinel.pipeline import export_palette


@pytest.fixture()
def client(mini_run, tmp_path):
    app = create_app(predictor=Predictor(mini_run, "cpu"), db_path=tmp_path / "t.db", storage_dir=tmp_path / "rep")
    return TestClient(app)


def _up(png, name="slice.png", ctype="image/png"):
    return {"file": (name, png, ctype)}


def test_health_and_model_info(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] is True and h["model_id"] == "patchcore_mini_s0" and "not a diagnosis" in h["disclaimer"]
    m = client.get("/api/v1/model").json()
    assert m["kind"] == "patchcore" and m["trained_on"] == "train/good" and m["thresholds"]["selected_on"] == "valid"


def test_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/api/v1/mri/analyze", "/api/v1/mri/quality-check", "/api/v1/jobs/{job_id}",
            "/api/v1/reports/{report_id}"} <= set(paths)


def test_analyze_persists_report_images_and_job(client, mini_png):
    r = client.post("/api/v1/mri/analyze", files=_up(mini_png["lesion"]))
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["outcome"] in ("no_region_flagged", "flagged_for_review", "uncertain_escalated") and a["human_review_status"] == "pending"
    rep = client.get(a["report_url"]).json()
    assert rep["inference"]["score"] == pytest.approx(a["score"]) and [t["node"] for t in rep["trace"]][0] == "validate_input"
    assert rep["job_id"] == a["job_id"] and rep["statements"] and "not a diagnosis" in rep["disclaimer"]
    for k in ("original", "heatmap", "overlay"):
        im = client.get(f"{a['report_url']}/images/{k}")
        assert im.status_code == 200 and im.headers["content-type"] == "image/png"
        assert Image.open(io.BytesIO(im.content)).size == (240, 240)
    job = client.get(f"/api/v1/jobs/{a['job_id']}").json()
    assert job["status"] == a["outcome"] and job["report_id"] == a["report_id"] and job["sha256"] == rep["input"]["sha256"]
    assert client.get("/api/v1/jobs").json()[0]["job_id"] == a["job_id"]


def test_identical_upload_gives_identical_score(client, mini_png):
    s = [client.post("/api/v1/mri/analyze", files=_up(mini_png["normal"])).json()["score"] for _ in range(2)]
    assert s[0] == pytest.approx(s[1])


def _rgb_photo():
    b = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 200, 3), dtype=np.uint8), "RGB").save(b, format="PNG")
    return b.getvalue()


@pytest.mark.parametrize("name,data,ctype,status,code", [
    ("a.png", b"", "image/png", 422, "empty_file"),
    ("a.png", b"not a png at all", "image/png", 422, "not_png"),
    ("a.txt", b"hello", "text/plain", 415, "not_png"),
    ("a.png", "photo", "image/png", 422, "wrong_size"),
    ("a.png", "huge", "image/png", 413, "file_too_large"),
    ("a.png", "trunc", "image/png", 422, "corrupt_image"),
])
def test_invalid_uploads(client, mini_png, name, data, ctype, status, code):
    data = {"photo": _rgb_photo(), "huge": b"\x89PNG\r\n\x1a\n" + b"0" * (MAX_BYTES + 10), "trunc": mini_png["normal"][:150]}.get(data, data)
    r = client.post("/api/v1/mri/analyze", files=_up(data, name, ctype))
    assert r.status_code == status and r.json()["detail"]["code"] == code
    if status == 422 and "job_id" in r.json()["detail"]:
        rep = client.get(f"/api/v1/reports/{r.json()['detail']['report_id']}").json()
        assert rep["outcome"]["status"] == "rejected_input" and rep["inference"] is None


def test_missing_file_field_is_422(client):
    assert client.post("/api/v1/mri/analyze").status_code == 422


def test_quality_check_endpoint_does_not_run_model(client, mini_png):
    r = client.post("/api/v1/mri/quality-check", files=_up(mini_png["normal"]))
    assert r.status_code == 200 and r.json()["quality"]["level"] in ("pass", "warn") and "inference" not in r.json()
    assert client.post("/api/v1/mri/quality-check", files=_up(b"zzz")).status_code == 422


def test_human_review_flow(client, mini_png):
    a = client.post("/api/v1/mri/analyze", files=_up(mini_png["normal"])).json()
    r = client.post(f"/api/v1/reports/{a['report_id']}/review", json={"decision": "annotate", "note": "check left side"})
    assert r.status_code == 200 and r.json()["human_review"]["status"] == "annotated"
    assert client.get(a["report_url"]).json()["human_review"]["note"] == "check left side"
    assert client.post(f"/api/v1/reports/{a['report_id']}/review", json={"decision": "diagnose"}).status_code == 422
    assert client.post("/api/v1/reports/nope/review", json={"decision": "accept"}).status_code == 404


def test_review_not_applicable_for_rejected_input(client):
    r = client.post("/api/v1/mri/analyze", files=_up(b"not a png"))
    rid = r.json()["detail"].get("report_id")
    assert rid and client.post(f"/api/v1/reports/{rid}/review", json={"decision": "accept"}).status_code == 409


def test_unknown_ids_404(client):
    for u in ("/api/v1/jobs/x", "/api/v1/reports/x", "/api/v1/reports/x/images/overlay"):
        assert client.get(u).status_code == 404
    assert client.get("/api/v1/reports/x/images/bad").status_code == 422


def test_concurrent_requests_are_serialised_and_consistent(client, mini_png):
    out = []

    def go():
        out.append(client.post("/api/v1/mri/analyze", files=_up(mini_png["lesion"])).json())
    ts = [threading.Thread(target=go) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(out) == 4 and len({round(o["score"], 6) for o in out}) == 1 and len({o["report_id"] for o in out}) == 4


def test_no_model_degrades_to_503_without_fabricating(mini_ctx, mini_png, tmp_path):
    export_palette(mini_ctx.cfg, tmp_path / "palette.json")
    app = create_app(predictor=None, model_dir=tmp_path / "does_not_exist", db_path=tmp_path / "t.db",
                     storage_dir=tmp_path / "r", palette_path=tmp_path / "palette.json")
    c = TestClient(app)
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] is False and h["model_id"] is None
    assert c.get("/api/v1/model").status_code == 503
    r = c.post("/api/v1/mri/analyze", files=_up(mini_png["normal"]))
    assert r.status_code == 503 and r.json()["detail"]["code"] == "no_model_available"
    assert c.post("/api/v1/mri/quality-check", files=_up(mini_png["normal"])).status_code == 200


def test_inference_failure_is_500_with_report_not_a_crash(mini_run, mini_png, tmp_path):
    class Boom(Predictor):
        def predict(self, idxs):
            raise RuntimeError("simulated")
    c = TestClient(create_app(predictor=Boom(mini_run, "cpu"), db_path=tmp_path / "t.db", storage_dir=tmp_path / "r"))
    r = c.post("/api/v1/mri/analyze", files=_up(mini_png["normal"]))
    assert r.status_code == 500 and r.json()["detail"]["code"] == "inference_failed"
    assert c.get(f"/api/v1/reports/{r.json()['detail']['report_id']}").json()["inference"] is None
    assert c.get("/health").status_code == 200


def test_llm_narrator_failure_does_not_block_response(mini_run, mini_png, tmp_path):
    def down(_):
        raise ConnectionError("provider unreachable")
    c = TestClient(create_app(predictor=Predictor(mini_run, "cpu"), db_path=tmp_path / "t.db", storage_dir=tmp_path / "r", narrator=down))
    r = c.post("/api/v1/mri/analyze", files=_up(mini_png["normal"]))
    assert r.status_code == 200
    assert c.get(r.json()["report_url"]).json()["llm_narrative"]["accepted"] is False


def test_default_device_env_override_and_fallback(monkeypatch):
    from brainsentinel.api.main import default_device
    monkeypatch.setenv("BRAINSENTINEL_DEVICE", "cpu")
    assert default_device() == "cpu"
    monkeypatch.delenv("BRAINSENTINEL_DEVICE")
    assert default_device() in ("cpu", "cuda")
