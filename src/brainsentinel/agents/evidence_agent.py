"""Nodes 4-9: model selection, detector, heatmap, evidence analysis, confidence, routing."""
from __future__ import annotations

import logging
import time

import numpy as np

from ..inference import evidence as ev
from ..inference.render import render_all
from .state import AgentState, step

log = logging.getLogger("brainsentinel.agent")
MAX_ATTEMPTS = 2   # one retry on inference failure


def make_select_model(predictor):
    def select_model(state: AgentState) -> dict:
        t0 = time.perf_counter()
        if predictor is None:
            return {"status": "no_model_available",
                    "errors": [{"code": "no_model_available", "message": "no trained model artifact is loaded"}],
                    "trace": [step("select_model", "fail", "no trained model artifact", t0)]}
        info = predictor.info
        return {"model_info": info, "trace": [step("select_model", "ok", f"{info['model_id']} ({info['kind']})", t0)]}
    return select_model


def make_run_detector(predictor):
    def run_detector(state: AgentState) -> dict:
        t0 = time.perf_counter()
        tr, last = [], None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                maps, scores, ms = predictor.predict([state["decoded"].idx])
                thr = predictor.thresholds or {}
                inf = {"score": float(scores[0]), "latency_ms": float(ms), "attempts": attempt,
                       "image_threshold": thr.get("image_threshold"), "pixel_threshold": thr.get("pixel_threshold"),
                       "threshold_selected_on": thr.get("selected_on")}
                tr.append(step("run_detector", "ok", f"score {inf['score']:.4f} in {ms:.1f} ms (attempt {attempt})", t0))
                return {"inference": inf, "amap": maps[0], "trace": tr}
            except Exception as e:  # noqa: BLE001 - any model failure must degrade, never crash the API
                last = f"{type(e).__name__}: {e}"
                log.warning("inference attempt %d failed: %s", attempt, last)
                tr.append(step("run_detector", "retry" if attempt < MAX_ATTEMPTS else "fail", last, t0))
        return {"status": "inference_failed", "errors": [{"code": "inference_failed", "message": last}], "trace": tr}
    return run_detector


def make_heatmap(predictor):
    def make_heatmap_node(state: AgentState) -> dict:
        t0 = time.perf_counter()
        rgb = predictor.palette.decode(state["decoded"].idx + 0)       # what the model saw (bg-corrected), palette-rendered
        imgs = render_all(rgb, state["amap"], state["inference"].get("pixel_threshold"))
        return {"images": imgs, "trace": [step("make_heatmap", "ok", imgs["display_scale"], t0)]}
    return make_heatmap_node


def make_analyze_evidence(predictor):
    def analyze_evidence(state: AgentState) -> dict:
        t0 = time.perf_counter()
        dec, amap, inf = state["decoded"], state["amap"], state["inference"]
        loc = ev.location_sanity(amap, dec.brain)
        regions = ev.flagged_regions(amap, inf.get("pixel_threshold"))
        try:
            pert = ev.perturbations(dec.idx)
            maps, scores, _ = predictor.predict([p[1] for p in pert])
            stab = ev.stability(amap, inf["score"], [(p[0], maps[i], float(scores[i]), p[2]) for i, p in enumerate(pert)], dec.brain)
            st = step("analyze_evidence", "ok", f"{stab['n_variants']} perturbations; mean map corr {stab['mean_map_corr']:.3f}", t0)
        except Exception as e:  # noqa: BLE001
            stab = None
            st = step("analyze_evidence", "fail", f"stability unavailable: {type(e).__name__}: {e}", t0)
        return {"stability": stab, "location": loc, "regions": regions, "trace": [st]}
    return analyze_evidence


def assess_confidence(state: AgentState) -> dict:
    t0 = time.perf_counter()
    inf = state["inference"]
    if state.get("stability") is None:
        e = {"flagged": None if inf.get("image_threshold") is None else inf["score"] >= inf["image_threshold"],
             "strength": "unavailable", "reasons": ["stability analysis could not be computed"], "margin_ratio": None}
    else:
        e = ev.assess_evidence(inf["score"], inf.get("image_threshold"), state["stability"], state["location"])
    return {"evidence": e, "trace": [step("assess_confidence", "ok", f"strength={e['strength']}", t0)]}


def route_decision(state: AgentState) -> str:
    """Conditional edge after confidence assessment (pure function of measured evidence)."""
    e = state["evidence"]
    if e["strength"] in ("weak", "unavailable") or (e["flagged"] is False and e["reasons"]):
        return "uncertain"
    return "flagged" if e["flagged"] else "clear"


def make_finalize(kind: str):
    status = {"clear": "no_region_flagged", "flagged": "flagged_for_review", "uncertain": "uncertain_escalated"}[kind]

    def finalize(state: AgentState) -> dict:
        return {"status": status, "route": kind, "trace": [step(f"route_{kind}", "route", f"-> {status}")]}
    finalize.__name__ = f"finalize_{kind}"
    return finalize
