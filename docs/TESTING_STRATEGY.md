# Testing strategy

Full suite in the authoring sandbox (CPU, no dataset): **138 passed, 74 skipped, 0 failed** (`pytest -q`). Skips: 71 need `data/raw/Brain` (existing Day 1-3 data-contract tests, to be run locally), 3 need CUDA (strict when a GPU exists).

| Area | File | Verifies |
|---|---|---|
| Metrics | `test_metrics.py` | vs scikit-learn, brute-force AUPRO, analytic random AUPRO, hand-computed Dice/IoU, thresholds, bootstrap |
| Models | `test_models.py` | shapes, determinism, localisation of an injected region, save/load, PaDiM covariance vs NumPy, PCA vs scikit-learn, coreset vs random |
| Protocol | `test_runner.py`, `test_pipeline.py` | train → freeze on valid → test once → rerun blocked; selection; PCA rule; CLI plumbing |
| Agent | `test_agent.py` | every branch, retry, grounding, LLM guards |
| API | `test_api.py` | endpoints, invalid uploads, persistence, concurrency, 503 degradation |
| Regression | `test_repo_hygiene.py`, `test_scripts_hygiene.py` | audit defects (README EOF, import order, output paths) |
| Frontend | `frontend/src/app.test.jsx` | 7 vitest tests; build verified |

**Fixtures are non-medical** (`tests/fixtures_nonmedical.py`, `fixtures_minidataset.py`): smooth fields / discs written only to pytest temp dirs. They validate code paths, never produce a reported metric. Model quality claims require running on the real data.

Not tested: real-data training/evaluation, GPU execution, Docker build, browser tests of every outcome state.
