# Day 9 - Model lock, inference service, final test

## 1. Original objectives
- Select on validation, freeze, single held-out test evaluation, predict_one, model card.

## 2. Implemented functionality
- Predictor, predict_one.py, select_model.py, test guard.

## 3. Files created or modified
- src/brainsentinel/inference/predict.py, scripts/predict_one.py

## 4. Technical review
- Inference path tested end to end on a fixture model, including CLI. Final test evaluation and model card require real runs.

## 5. Tests executed
- test_agent.py::test_predict_one_cli, test_api.py. Final test metrics: DONE (once per model). Model card: written from measured numbers (docs/MODEL_CARD.md).

## 6. Actual test results
- Selection (validation only, `results/model_selection.json`): patchcore_s0, mean rank 1.0 on all three ranking metrics; all candidates passed the VRAM/latency gates. `artifacts/selected_model.json` points to it. `evaluations.jsonl` records exactly one test evaluation per model, none with `--allow-rerun`; validation was re-evaluated once for the autoencoder and PaDiM (allowed).
- Backend served the selected model; the UI produced a report on the user's machine.

## 7. Bugs discovered
- -

## 8. Fixes applied
- -

## 9. Remaining issues
- Multi-seed confirmation; model-card review.

## 10. Acceptance decision
- CONDITIONAL PASS (model locked, test run once, card written; single seed).

## 11. Evidence and artifact references
- artifacts/selected_model.json, results/model_selection.json, artifacts/models/patchcore_s0/{threshold.json,evaluations.jsonl,metrics_test.json}, docs/MODEL_CARD.md
