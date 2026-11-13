# API documentation (interactive docs at `/docs`)

| Method & path | Purpose | Errors |
|---|---|---|
| `GET /health` | status, model_loaded, model_id, device, disclaimer | - |
| `GET /api/v1/model` | model kind, hparams, thresholds, validation summary | 503 no model |
| `POST /api/v1/mri/quality-check` (`file`) | domain + quality only, no model | 413/415/422 |
| `POST /api/v1/mri/analyze` (`file`) | run the agent graph; returns summary + `report_url` | 413 too large, 415 not .png, 422 rejected input, 503 no model, 500 inference_failed |
| `GET /api/v1/jobs`, `/jobs/{id}` | job history | 404 |
| `GET /api/v1/reports/{id}` | full report incl. statements, evidence, trace | 404 |
| `GET /api/v1/reports/{id}/images/{original\|heatmap\|overlay}` | PNGs from the stored run | 404 |
| `POST /api/v1/reports/{id}/review` `{decision: accept\|reject\|annotate, note?}` | human review | 404, 409 not applicable, 422 |

Error body: `{"detail": {"code", "message", "job_id"?, "report_id"?}}`. Rejected and failed jobs still produce a stored report. Uploads: `.png` ≤ 2 MB. One inference at a time (lock). Model: `BRAINSENTINEL_MODEL_DIR` or `artifacts/selected_model.json`; without one the service runs and answers 503 instead of fabricating output.

```bash
curl -F "file=@slice.png;type=image/png" localhost:8000/api/v1/mri/analyze
```
