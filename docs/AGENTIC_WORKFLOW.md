# Agentic workflow (LangGraph)

Nodes call deterministic code; none free-writes findings. Routing depends on measured values.

| Node | Tool it calls | Terminal status on failure |
|---|---|---|
| validate_input | `decode_upload` (size, PNG magic, corrupt, opaque RGBA, palette membership) | `rejected_input` |
| verify_domain | conformance to training input format (NOT a modality classifier) | `domain_mismatch` |
| assess_quality | brain fraction, texture (Laplacian var), saturation, background offset | `quality_failed` (model not run) |
| select_model | loaded Predictor + frozen thresholds | `no_model_available` |
| run_detector | `Predictor.predict`, **one retry** | `inference_failed` |
| make_heatmap | fixed display scale (0..2 x validation pixel threshold) | - |
| analyze_evidence | 7 deterministic perturbations (±3% intensity, ±2 px shifts, σ=2 noise), location sanity, flagged regions | stability `None` → `unavailable` |
| assess_confidence | explicit rules (`inference/evidence.py::EVIDENCE_RULES`) | - |
| route | `clear` / `flagged` / `uncertain` | - |
| build_report | templated statements, each with `refs` that must resolve | - |
| human_review | status `pending` (or `not_applicable` for rejected input) | - |

Evidence strength: `not_flagged`; `strong` (flagged, margin ≥ 25%, no instability/artefact signal); `moderate`; `weak` (any instability or artefact reason) → escalated as `uncertain_escalated`. It is a **heuristic review signal, not a calibrated confidence**.

LLM: optional narrator hook, **off by default, no provider shipped**. Its text is dropped if it contains clinical wording, numbers absent from the evidence, or is too long; if it raises, results are unaffected. It is labelled "not part of the evidence".

Unsupported inputs (photos, other sizes, non-palette renderings) are rejected before any model runs. Limitation: a palette-conformant image of another MRI sequence cannot be detected.
