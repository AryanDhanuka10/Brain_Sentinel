# Day 6 - PatchCore

## 1. Original objectives
- PatchCore with ResNet18, coreset, 8 GB feasibility, validation metrics.

## 2. Implemented functionality
- models/patchcore_model.py (3x3 aggregation, greedy k-center on random projection, 1-NN); configs/patchcore.yaml (max 2,000 train slices, 1% coreset).

## 3. Files created or modified
- src/brainsentinel/models/patchcore_model.py, configs/patchcore.yaml

## 4. Technical review
- Coreset beats random sampling in coverage and is seeded-deterministic (tested). Measured on the user's GPU: train peak VRAM 3.46 GB, coreset selection 404 s of 411 s total. Deviation: ≤2,000 random train slices; no score re-weighting.

## 5. Tests executed
- test_models.py (coreset, bank dim 384, PCA variant). Real-data run: DONE.

## 6. Actual test results
- PatchCore (real data, seed 0, 2,000 train slices, 1% coreset, bank 18,000 x 384): 410.8 s (coreset 404 s), 37.0 MB, train peak VRAM 3.46 GB (fits 8 GB), inference VRAM 0.13 GB, latency 4.5 ms. Validation: image AUROC 0.930, pixel AUROC 0.985, AUPRO 0.898, Dice 0.520. Test (once): image AUROC 0.909 [0.884, 0.931], pixel AUROC 0.979 full / 0.922 brain ROI, AUPRO 0.847 / 0.583, Dice per-image 0.501 (pooled 0.600); sensitivity 0.789, specificity 0.878 (FPR 0.122). Re-training the identical config (patchcore_nopca_s0) reproduced the validation metrics exactly.

## 7. Bugs discovered
- -

## 8. Fixes applied
- -

## 9. Remaining issues
- Multi-seed runs not done. Brain-ROI AUPRO (0.583) is much lower than full-frame AUPRO (0.847): localisation inside the brain is the weak point.

## 10. Acceptance decision
- PASS for the stage (trained, measured, logged, reproducible); single seed.

## 11. Evidence and artifact references
- artifacts/models/patchcore_s0/*, results/logs/train_patchcore.txt, eval_valid_patchcore_s0.txt, eval_test.txt, docs/RESULTS.md
