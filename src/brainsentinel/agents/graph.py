"""LangGraph assembly.

validate_input -> verify_domain -> assess_quality -> select_model -> run_detector -> make_heatmap
 -> analyze_evidence -> assess_confidence -> {route_clear | route_flagged | route_uncertain}
 -> build_report -> human_review -> END
Any failing stage jumps straight to build_report with a terminal status (no model output is invented).
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .evidence_agent import (assess_confidence, make_analyze_evidence, make_finalize, make_heatmap,
                             make_run_detector, make_select_model, route_decision)
from .quality_agent import assess_quality_node, make_validate_input, verify_domain
from .report_agent import build_report, human_review_gate
from .state import AgentState


def _ok(state: AgentState) -> str:
    return "stop" if state.get("status") else "go"


def build_graph(predictor, palette, narrator=None):
    """Compile the workflow. ``predictor`` may be None (then select_model routes to no_model_available)."""
    g = StateGraph(AgentState)
    g.add_node("validate_input", make_validate_input(palette))
    g.add_node("verify_domain", verify_domain)
    g.add_node("assess_quality", assess_quality_node)
    g.add_node("select_model", make_select_model(predictor))
    g.add_node("run_detector", make_run_detector(predictor))
    g.add_node("make_heatmap", make_heatmap(predictor))
    g.add_node("analyze_evidence", make_analyze_evidence(predictor))
    g.add_node("assess_confidence", assess_confidence)
    for k in ("clear", "flagged", "uncertain"):
        g.add_node(f"route_{k}", make_finalize(k))
    g.add_node("build_report", build_report(narrator))
    g.add_node("human_review", human_review_gate)
    g.set_entry_point("validate_input")
    for a, b in (("validate_input", "verify_domain"), ("verify_domain", "assess_quality"),
                 ("assess_quality", "select_model"), ("select_model", "run_detector"), ("run_detector", "make_heatmap")):
        g.add_conditional_edges(a, _ok, {"go": b, "stop": "build_report"})
    g.add_edge("make_heatmap", "analyze_evidence")
    g.add_edge("analyze_evidence", "assess_confidence")
    g.add_conditional_edges("assess_confidence", route_decision,
                            {"clear": "route_clear", "flagged": "route_flagged", "uncertain": "route_uncertain"})
    for k in ("clear", "flagged", "uncertain"):
        g.add_edge(f"route_{k}", "build_report")
    g.add_edge("build_report", "human_review")
    g.add_edge("human_review", END)
    return g.compile()


def run_analysis(graph, data: bytes, filename: str = "upload.png") -> dict:
    """Run the graph and return the final state."""
    return graph.invoke({"data": data, "filename": filename, "errors": [], "trace": []})
