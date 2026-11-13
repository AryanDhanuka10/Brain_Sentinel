"""FastAPI service. Results shown by any client are exactly what the agent graph computed and persisted here.

Run:  uvicorn brainsentinel.api.main:app_factory --factory --port 8000      (docs at /docs)
Model: BRAINSENTINEL_MODEL_DIR or artifacts/selected_model.json. With no trained model the service stays up,
/health reports model_loaded=false and /analyze answers 503 (it never fabricates a result).
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .. import __version__
from ..agents.graph import build_graph, run_analysis
from ..agents.state import DISCLAIMER
from ..inference import InputRejected, Predictor, assess_quality, decode_upload, resolve_model_dir
from ..pipeline import load_palette
from .db import Store

REPO = Path(__file__).resolve().parents[3]
log = logging.getLogger("brainsentinel.api")
MAX_BYTES = 2 * 1024 * 1024
HTTP_FOR_STATUS = {"rejected_input": 422, "domain_mismatch": 422, "no_model_available": 503, "inference_failed": 500}


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    model_loaded: bool
    model_id: str | None = None
    device: str | None = None
    disclaimer: str = DISCLAIMER


class ReviewIn(BaseModel):
    decision: Literal["accept", "reject", "annotate"]
    note: str | None = Field(None, max_length=2000)


class AnalyzeOut(BaseModel):
    job_id: str
    report_id: str
    outcome: str
    flagged: bool | None
    verdict: str | None = None
    score: float | None
    image_threshold: float | None
    evidence_strength: str | None
    human_review_status: str
    report_url: str
    disclaimer: str = DISCLAIMER


def create_app(predictor: Predictor | None = None, model_dir: str | Path | None = None, db_path: str | Path | None = None,
               storage_dir: str | Path | None = None, device: str = "cpu", narrator=None, palette_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="BrainSentinel API", version=__version__,
                  description="Research prototype: brain MRI anomaly localisation with an evidence-grounded review workflow. "
                              + DISCLAIMER)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    store = Store(db_path or REPO / "artifacts" / "brainsentinel.db")
    storage = Path(storage_dir or REPO / "artifacts" / "reports")
    storage.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()                                       # one inference at a time (single small GPU)
    if predictor is None:
        md = Path(model_dir) if model_dir else resolve_model_dir(REPO)
        if md is not None:
            try:
                predictor = Predictor(md, device)
            except Exception as e:  # noqa: BLE001
                log.error("could not load model from %s: %s", md, e)
    palette = predictor.palette if predictor else None
    if palette is None:
        pp = Path(palette_path or REPO / "artifacts" / "palette.json")
        palette = load_palette(pp) if pp.exists() else None
    graph = build_graph(predictor, palette, narrator) if palette is not None else None
    app.state.predictor, app.state.store = predictor, store

    async def read_png(file: UploadFile) -> tuple[bytes, str]:
        name = file.filename or "upload.png"
        if not name.lower().endswith(".png"):
            raise HTTPException(415, {"code": "not_png", "message": "Only .png files are accepted."})
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, {"code": "file_too_large", "message": f"Maximum upload is {MAX_BYTES // 2**20} MB."})
        return data, name

    @app.get("/health", response_model=Health)
    def health():
        return Health(version=__version__, model_loaded=predictor is not None,
                      model_id=predictor.model_id if predictor else None, device=predictor.device if predictor else None)

    @app.get("/api/v1/model")
    def model_info():
        if predictor is None:
            raise HTTPException(503, {"code": "no_model_available", "message": "No trained model artifact is loaded."})
        return predictor.info

    @app.post("/api/v1/mri/quality-check")
    async def quality_check(file: UploadFile = File(...)):
        if palette is None:
            raise HTTPException(503, {"code": "palette_unavailable", "message": "artifacts/palette.json is missing (run scripts/export_palette.py)."})
        data, name = await read_png(file)
        try:
            s = decode_upload(data, palette)
        except InputRejected as e:
            raise HTTPException(422, {"code": e.code, "message": e.message}) from e
        return {"filename": name, "domain": dict(s.domain, sha256=s.sha256), "quality": assess_quality(s), "disclaimer": DISCLAIMER}

    @app.post("/api/v1/mri/analyze", response_model=AnalyzeOut)
    async def analyze(file: UploadFile = File(...)):
        if graph is None:
            raise HTTPException(503, {"code": "palette_unavailable", "message": "artifacts/palette.json is missing."})
        data, name = await read_png(file)
        job_id = uuid.uuid4().hex[:12]
        store.new_job(job_id, name)
        t0 = time.perf_counter()
        try:
            with lock:
                st = run_analysis(graph, data, name)
        except Exception as e:  # noqa: BLE001 - graph nodes already degrade; this is the last line of defence
            log.exception("graph crashed")
            store.finish_job(job_id, "error", error_code="internal_error")
            raise HTTPException(500, {"code": "internal_error", "message": f"{type(e).__name__}", "job_id": job_id}) from e
        rep = st["report"]
        rep["trace"], rep["job_id"] = st["trace"], job_id
        if st.get("images"):
            d = storage / rep["report_id"]
            d.mkdir(parents=True, exist_ok=True)
            for k in ("original", "heatmap", "overlay"):
                (d / f"{k}.png").write_bytes(st["images"][k])
        store.save_report(job_id, rep)
        store.finish_job(job_id, st["status"], rep["report_id"], rep["input"]["sha256"],
                         (st.get("errors") or [{}])[0].get("code"), (time.perf_counter() - t0) * 1000)
        code = HTTP_FOR_STATUS.get(st["status"])
        if code:
            err = (st.get("errors") or [{"code": st["status"], "message": st["status"]}])[0]
            return JSONResponse(status_code=code, content={"detail": {**err, "job_id": job_id, "report_id": rep["report_id"]}})
        inf, e = rep.get("inference") or {}, rep.get("evidence") or {}
        return AnalyzeOut(job_id=job_id, report_id=rep["report_id"], outcome=st["status"], flagged=e.get("flagged"), verdict=(rep.get("verdict") or {}).get("label"),
                          score=inf.get("score"), image_threshold=inf.get("image_threshold"),
                          evidence_strength=e.get("strength"), human_review_status=rep["human_review"]["status"],
                          report_url=f"/api/v1/reports/{rep['report_id']}")

    @app.get("/api/v1/jobs")
    def list_jobs(limit: int = 50):
        return store.jobs(max(1, min(limit, 200)))

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str):
        j = store.job(job_id)
        if not j:
            raise HTTPException(404, {"code": "job_not_found", "message": job_id})
        return j

    @app.get("/api/v1/reports/{report_id}")
    def get_report(report_id: str):
        r = store.report(report_id)
        if not r:
            raise HTTPException(404, {"code": "report_not_found", "message": report_id})
        return r

    @app.get("/api/v1/reports/{report_id}/images/{name}")
    def get_image(report_id: str, name: Literal["original", "heatmap", "overlay"]):
        if not report_id.isalnum():
            raise HTTPException(404, {"code": "report_not_found", "message": report_id})
        p = storage / report_id / f"{name}.png"
        if not p.exists():
            raise HTTPException(404, {"code": "image_not_found", "message": f"{name} not available for this report"})
        return Response(p.read_bytes(), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/api/v1/reports/{report_id}/review")
    def review(report_id: str, body: ReviewIn):
        r = store.report(report_id)
        if not r:
            raise HTTPException(404, {"code": "report_not_found", "message": report_id})
        if r["human_review"]["status"] == "not_applicable":
            raise HTTPException(409, {"code": "review_not_applicable", "message": "Input was rejected; there is nothing to review."})
        return store.add_review(report_id, body.decision, body.note)

    @app.exception_handler(HTTPException)
    async def http_exc(_: Request, exc: HTTPException):
        d = exc.detail if isinstance(exc.detail, dict) else {"code": "http_error", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content={"detail": d})

    return app


def default_device() -> str:
    """BRAINSENTINEL_DEVICE (cpu|cuda|cuda:0) if set, else CUDA when available, else CPU."""
    import os

    import torch
    return os.environ.get("BRAINSENTINEL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def app_factory() -> FastAPI:
    return create_app(device=default_device())
