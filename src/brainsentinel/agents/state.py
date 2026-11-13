"""Typed agent state. ``trace`` and ``errors`` accumulate across nodes (operator.add reducer)."""
from __future__ import annotations

import operator
import time
from typing import Annotated, Any, TypedDict

DISCLAIMER = ("Research prototype. Output is an anomaly score and map from a model trained on normal slices; "
              "it flags regions for review by a qualified human reviewer. It is not a diagnosis, not clinically "
              "validated, and an anomaly score is not a disease probability.")


class AgentState(TypedDict, total=False):
    filename: str
    data: bytes
    decoded: Any                      # inference.checks.DecodedSlice
    domain: dict
    quality: dict
    model_info: dict
    inference: dict
    amap: Any                         # np.ndarray (not serialised into the report)
    images: dict                      # PNG bytes: original/heatmap/overlay + display_scale
    stability: dict
    location: dict
    regions: list
    evidence: dict
    status: str                       # terminal outcome, see STATUSES
    route: str
    report: dict
    human_review: dict
    errors: Annotated[list, operator.add]
    trace: Annotated[list, operator.add]


STATUSES = ("rejected_input", "domain_mismatch", "quality_failed", "no_model_available", "inference_failed",
            "no_region_flagged", "flagged_for_review", "uncertain_escalated")


def step(node: str, status: str, detail: str = "", t0: float | None = None, **extra) -> dict:
    """One trace row. ``status`` is ok | fail | skip | retry | route."""
    return {"node": node, "status": status, "detail": detail,
            "ms": round((time.perf_counter() - t0) * 1000, 2) if t0 else None, **extra}
