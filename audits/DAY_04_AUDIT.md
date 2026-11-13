# Day 4 - Autoencoder baseline

## 1. Original objectives
- Normal-only conv autoencoder, residual maps, validation metrics, saved results.

## 2. Implemented functionality
- models/autoencoder.py; train.py/evaluate.py; configs/autoencoder.yaml; early stopping on valid/good normals only.

## 3. Files created or modified
- src/brainsentinel/models/autoencoder.py, common.py, pipeline.py, scripts/train.py, scripts/evaluate.py, configs/autoencoder.yaml

## 4. Technical review
- Loss decreases; same-size maps; deterministic predict; save/load identical; checkpoint selection never sees abnormal slices or masks.

## 5. Tests executed
- test_models.py + test_pipeline.py pass in the sandbox; on the user's machine `pytest -q` gave 212 passed, 0 failed (74 data/GPU tests now executed).
- Real-data training: DONE (`python scripts/train.py --model autoencoder`).

## 6. Actual test results
- Autoencoder (real data, seed 0): trained 2026-11-10, 642.9 s, 5.3 MB, train peak VRAM 0.30 GB, latency 0.69 ms. Validation: image AUROC 0.352, pixel AUROC 0.832, AUPRO 0.502, Dice 0.193. Test (evaluated once): image AUROC 0.254 [0.217, 0.290], pixel AUROC 0.854 full frame but **0.497 inside the brain ROI (chance)**, AUPRO 0.592 / 0.210 (full / ROI); at the validation-frozen image threshold sensitivity 0.996 and specificity 0.000, i.e. it flags every slice. It is a failed baseline: image-level scores are inverted and the pixel score only separates background. Cause not diagnosed.

## 7. Bugs discovered
- First real run crashed: `scripts/train.py` injected a `seed` hyper-parameter the autoencoder does not accept (`TypeError`). Logged as FAILED in `results/experiment_log.csv` (honest audit trail, left in place).
- Image-level scores inverted (AUROC 0.254 on test); pixel scores carry no signal inside the brain (ROI AUROC 0.497).

## 8. Fixes applied
- `train.py` now sets `seed` only for models that declare it; regression test added. Rerun succeeded.

## 9. Remaining issues
- Diagnose the inverted image score (hypothesis only: top-1% reconstruction error tracks anatomy/slice position, not lesions). Report as a failed baseline; do not tune on test.

## 10. Acceptance decision
- CONDITIONAL PASS (stage executed and logged; the model itself is not usable: specificity 0.000 at the frozen threshold).

## 11. Evidence and artifact references
- artifacts/models/autoencoder_s0/{config,threshold,metrics_valid,metrics_test}.json, evaluations.jsonl, train_log.csv
- results/logs/train_autoencoder.txt, results/experiment_log.csv, docs/RESULTS.md
- tests/test_models.py, tests/test_pipeline.py
