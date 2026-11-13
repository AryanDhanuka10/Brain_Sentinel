# Master development tracker

PASS = all mandatory criteria executed and verified; CONDITIONAL PASS = core works, non-blocking issues remain; BLOCKED = needs your local dataset/GPU run. Nothing is PASS merely because code exists.

| Day | Milestone | Status | Code complete | Real-data run |
|---|---|---|---|---|
| 00 | Specification and repository reconciliation (retrospective) | **PASS** | n/a | n/a |
| 01 | Research, setup, dataset download (retrospective) | **CONDITIONAL PASS** | yes | done by user |
| 02 | Dataset audit and visualisation (retrospective) | **CONDITIONAL PASS** | yes (notebook/montage missing) | done by user |
| 03 | Preprocessing and tests (retrospective) | **CONDITIONAL PASS** | yes (montage missing) | done by user |
| 04 | Autoencoder baseline | **CONDITIONAL PASS** | yes | done 2026-11-10 (model fails as baseline) |
| 05 | PaDiM | **CONDITIONAL PASS** | yes | done (paper-default config tested) |
| 06 | PatchCore | **PASS** | yes | done (single seed) |
| 07 | Fair comparison and error analysis | **CONDITIONAL PASS** | yes (error-analysis plots missing) | done |
| 08 | PCA and preprocessing ablations | **CONDITIONAL PASS** | partly (ablations 2/3 not implemented) | PCA done (validation only) |
| 09 | Model lock, inference service, final test | **CONDITIONAL PASS** | yes | done (selected PatchCore; test once) |
| 10 | LangGraph agentic workflow | **CONDITIONAL PASS** | yes | n/a |
| 11 | FastAPI and frontend | **CONDITIONAL PASS** | yes | n/a |
| 12 | Integration, report, UX tests | **CONDITIONAL PASS** | yes | one manual real-model run |
| 13 | Final evaluation, demo, slides | **BLOCKED** | outline only | results ready; slides/demo owed |
| 14 | Submission freeze | **BLOCKED** | n/a | n/a |
