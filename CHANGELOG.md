# Changelog

## 0.1.0 - Days 4-14 implementation (2026-11-08)
Added: evaluation framework, three models + PCA reducer, pipeline/CLI scripts, predictor, LangGraph agent, FastAPI + SQLite, React UI, Docker files, docs, audits, tests (138 pass in sandbox).
Fixed (Day 0-3 audit): data/README.md stray `EOF` + wrong path; empty pyproject.toml; Day-1 hardware tests not CPU-safe; build_index.py import-before-sys.path crash; Day-2 scripts wrote to results/ root; misleading `images_with_nonblack_corner` key; stale TBD in day3_notes.md; requirements.txt missing later dependencies.
Known open: notebooks/montages, Docker unverified, real-data results NOT RUN (see audits/MASTER_TRACKER.md).

## 0.1.1 (2026-11-13)
- Real-data runs recorded: three models trained/evaluated, PatchCore selected, PCA experiments, `docs/RESULTS.md`, `docs/MODEL_CARD.md`; audits updated with measured results.
- Fixed: `train.py` seed hyper-parameter crash (autoencoder); interrupted training logged as OK; `pca_experiments.py` `nopca` tag bug; API serving device default (CUDA if available, `BRAINSENTINEL_DEVICE`).
- Added `scripts/summarize_results.py`.
- Fixed test isolation: `train_run` copied `artifacts/palette.json` into every run even when the training context had a different palette (17 test failures once the real palette existed). It now copies it only if identical to the context palette; regression test added.
- Added a plain `verdict` (ANOMALY_DETECTED / NO_ANOMALY_DETECTED / INCONCLUSIVE, with borderline/low-reliability flag) to reports, the API response and the UI headline; human review remains an optional confirmation step.
