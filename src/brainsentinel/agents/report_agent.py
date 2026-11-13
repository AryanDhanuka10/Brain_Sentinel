"""Node 10-11: structured review report (deterministic, grounded) and the human-review gate."""
from __future__ import annotations

import time
import uuid

from .narrate import safe_narrative
from .state import DISCLAIMER, AgentState, step

LIMITATIONS = [
    "Trained and validated only on BMAD Brain (BraTS 2021 FLAIR) 2D slices; behaviour on other scanners, sequences or populations is unknown.",
    "Anomaly scores are model-specific and not calibrated probabilities; a high score can result from artefacts, normal anatomical variation or acquisition differences.",
    "The domain check verifies input-format conformance only; it cannot confirm the imaging sequence.",
    "Evidence strength is a heuristic review signal derived from explicit rules, not a clinical confidence.",
]

OUTCOME_TEXT = {
    "no_region_flagged": "No region was flagged: the image anomaly score is below the validation-frozen threshold.",
    "flagged_for_review": "A region is flagged for human review: the image anomaly score is at or above the validation-frozen threshold and the evidence checks found no instability.",
    "uncertain_escalated": "The result is uncertain and escalated to a human reviewer: the evidence checks raised the concerns listed below.",
    "rejected_input": "The input was rejected and no model was run.",
    "domain_mismatch": "The input does not match the model's training input format and no model was run.",
    "quality_failed": "The image failed the deterministic quality gate and no model was run; please re-upload or review manually.",
    "no_model_available": "No trained model artifact is available, so no analysis was performed.",
    "inference_failed": "The model failed to produce a result; no anomaly output is available.",
}


BORDERLINE_FRAC = 0.05          # heuristic: scores within 5% of the frozen threshold are called borderline


def build_verdict(r: dict) -> dict:
    """Plain model verdict from the measured score vs the validation-frozen threshold (deterministic, no new numbers)."""
    inf, status = r.get("inference"), r["outcome"]["status"]
    thr = (inf or {}).get("image_threshold")
    if not inf or thr is None or status not in ("no_region_flagged", "flagged_for_review", "uncertain_escalated"):
        return {"label": "INCONCLUSIVE", "borderline": False, "reliability": "none",
                "text": "No model verdict: the analysis did not complete (see the outcome above)."}
    above = inf["score"] >= thr
    borderline = abs(inf["score"] - thr) <= BORDERLINE_FRAC * abs(thr)
    low = status == "uncertain_escalated" or borderline
    if above:
        text = "Model verdict: ANOMALY DETECTED - this slice differs from the normal slices the model was trained on."
    else:
        text = "Model verdict: NO ANOMALY DETECTED - this slice resembles the normal slices the model was trained on."
    if low:
        text += " Low reliability: the score is close to the threshold or the stability checks were not conclusive."
    return {"label": "ANOMALY_DETECTED" if above else "NO_ANOMALY_DETECTED", "borderline": bool(borderline),
            "reliability": "low" if low else "normal", "text": text}


def _get(d: dict, path: str):
    for p in path.split("."):
        d = d[int(p)] if isinstance(d, list) else d[p]
    return d


def build_statements(r: dict) -> list[dict]:
    """Every statement carries ``refs`` (dotted paths into the report) from which its numbers are taken."""
    S = []

    def add(text, *refs):
        S.append({"text": text, "refs": list(refs)})
    if r.get("verdict"):
        add(r["verdict"]["text"], "verdict.label", "verdict.reliability")
    add(OUTCOME_TEXT[r["outcome"]["status"]], "outcome.status")
    if r.get("input", {}).get("filename"):
        add(f"Input file '{r['input']['filename']}' (sha256 {r['input']['sha256'][:12]}...).", "input.filename", "input.sha256")
    for e in r.get("errors", []):
        add(f"Error {e['code']}: {e['message']}", "errors")
    if r.get("quality"):
        q = r["quality"]
        add(f"Quality gate: {q['level']}; foreground fraction {q['metrics']['brain_fraction']:.3f}.", "quality.level", "quality.metrics.brain_fraction")
    inf = r.get("inference")
    if inf:
        t = inf.get("image_threshold")
        if t is not None:
            add(f"Image anomaly score {inf['score']:.4f} versus validation-frozen threshold {t:.4f} (model {r['model']['model_id']}).",
                "inference.score", "inference.image_threshold", "model.model_id")
        else:
            add(f"Image anomaly score {inf['score']:.4f}; no validation-frozen threshold exists for this model, so no flag is made.",
                "inference.score")
        add(f"Inference took {inf['latency_ms']:.1f} ms.", "inference.latency_ms")
    ev = r.get("evidence")
    if ev:
        add(f"Evidence strength (heuristic): {ev['strength']}.", "evidence.strength")
        for i, reason in enumerate(ev["reasons"]):
            add(f"Evidence note: {reason}.", f"evidence.reasons.{i}")
    st = r.get("stability")
    if st:
        add(f"Across {st['n_variants']} small perturbations the mean map correlation is {st['mean_map_corr']:.3f} "
            f"and the image score varies with CV {st['score_cv']:.3f}.", "stability.n_variants", "stability.mean_map_corr", "stability.score_cv")
    for i, g in enumerate(r.get("regions", [])[:3]):
        y0, x0, y1, x1 = g["bbox_yxyx"]
        add(f"Region {i + 1}: {g['area_px']} px above the pixel threshold, bounding box rows {y0}-{y1}, columns {x0}-{x1}, "
            f"peak score {g['max_score']:.4f}.", f"regions.{i}.area_px", f"regions.{i}.bbox_yxyx", f"regions.{i}.max_score")
    return S


def build_report(deps_narrator=None):
    def build_report_node(state: AgentState) -> dict:
        t0 = time.perf_counter()
        dec = state.get("decoded")
        r = {"report_id": uuid.uuid4().hex[:12], "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "disclaimer": DISCLAIMER,
             "outcome": {"status": state["status"], "route": state.get("route")},
             "input": {"filename": state.get("filename"), "sha256": (state.get("domain") or {}).get("sha256", "n/a"),
                       "n_bytes": (state.get("domain") or {}).get("n_bytes")},
             "errors": state.get("errors", []), "domain": state.get("domain"), "quality": state.get("quality"),
             "model": state.get("model_info"), "inference": state.get("inference"), "evidence": state.get("evidence"),
             "stability": state.get("stability"), "location": state.get("location"), "regions": state.get("regions", []),
             "limitations": LIMITATIONS}
        if state.get("images"):
            r["display_scale"] = state["images"]["display_scale"]
        r["verdict"] = build_verdict(r)
        r["statements"] = build_statements(r)
        for s in r["statements"]:                      # grounding self-check: every ref must resolve
            for ref in s["refs"]:
                _get(r, ref)
        r["llm_narrative"] = None
        if deps_narrator is not None:
            try:
                txt, why = safe_narrative(deps_narrator(" ".join(s["text"] for s in r["statements"])),
                                          " ".join(s["text"] for s in r["statements"]))
                r["llm_narrative"] = {"text": txt, "accepted": txt is not None, "reason": why,
                                      "label": "language-model text, not part of the measured evidence"}
            except Exception as e:  # noqa: BLE001 - LLM failure must not affect results
                r["llm_narrative"] = {"text": None, "accepted": False, "reason": f"narrator failed: {type(e).__name__}"}
        return {"report": r, "trace": [step("build_report", "ok", f"{len(r['statements'])} grounded statements", t0)]}
    return build_report_node


def human_review_gate(state: AgentState) -> dict:
    """Every analysed result waits for a human decision; rejected inputs have nothing to review."""
    if state["status"] in ("rejected_input", "domain_mismatch"):
        hr = {"status": "not_applicable", "reason": "input rejected before analysis"}
    else:
        hr = {"status": "pending", "options": ["accept", "reject", "annotate"], "note": None, "decided_at": None}
    rep = dict(state["report"], human_review=hr)
    return {"human_review": hr, "report": rep, "trace": [step("human_review", "ok", hr["status"])]}
