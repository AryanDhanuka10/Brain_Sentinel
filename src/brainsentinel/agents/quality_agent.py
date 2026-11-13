"""Nodes 1-3: input validation, imaging-domain verification, image-quality assessment (all deterministic)."""
from __future__ import annotations

import time

from ..inference.checks import InputRejected, assess_quality, decode_upload
from .state import AgentState, step


def make_validate_input(palette):
    def validate_input(state: AgentState) -> dict:
        t0 = time.perf_counter()
        try:
            dec = decode_upload(state["data"], palette)
        except InputRejected as e:
            return {"status": "rejected_input", "errors": [{"code": e.code, "message": e.message}],
                    "trace": [step("validate_input", "fail", f"{e.code}: {e.message}", t0)]}
        return {"decoded": dec, "domain": dict(dec.domain, sha256=dec.sha256, n_bytes=dec.n_bytes),
                "trace": [step("validate_input", "ok", "opaque 240x240 RGBA PNG; every colour in the training palette", t0)]}
    return validate_input


def verify_domain(state: AgentState) -> dict:
    """Conformance to the training input format (NOT a modality classifier). See inference.checks docstring."""
    t0 = time.perf_counter()
    d = state["domain"]
    ok = bool(d.get("palette_conformant")) and d.get("width") == d.get("height") == 240
    d = dict(d, declared_domain="BMAD Brain FLAIR axial slice, 240x240, palette-encoded",
             note="conformance check only; the sequence/modality of a conforming image cannot be verified here")
    if not ok:
        return {"domain": d, "status": "domain_mismatch",
                "errors": [{"code": "domain_mismatch", "message": "input does not match the training input format"}],
                "trace": [step("verify_domain", "fail", "not conformant", t0)]}
    return {"domain": d, "trace": [step("verify_domain", "ok", "conforms to the model's training input format", t0)]}


def assess_quality_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    q = assess_quality(state["decoded"])
    if q["level"] == "fail":
        codes = ",".join(f["code"] for f in q["flags"] if f["level"] == "fail")
        return {"quality": q, "status": "quality_failed",
                "errors": [{"code": "quality_failed", "message": f"quality gate failed: {codes}"}],
                "trace": [step("assess_quality", "fail", codes, t0)]}
    return {"quality": q, "trace": [step("assess_quality", q["level"], ",".join(f["code"] for f in q["flags"]) or "no flags", t0)]}
