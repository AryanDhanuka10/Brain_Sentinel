# Day 5 - PaDiM

## 1. Original objectives
- PaDiM on a light backbone, validation metrics, memory/runtime record.

## 2. Implemented functionality
- models/padim_model.py (streamed float64 moments with shift, regularised inverse, Mahalanobis, upsample+blur); configs/padim.yaml.

## 3. Files created or modified
- src/brainsentinel/models/padim_model.py, features.py, reducers.py, configs/padim.yaml

## 4. Technical review
- Covariance and mean match NumPy; fit refuses <2 images; non-finite stats raise; unfitted predict raises. Memory: d=100 ≈ 72 MB moments; d=384 ≈ 1.1 GB float64 (device).

## 5. Tests executed
- test_models.py: shapes at 240x240 (mean (900,100)), covariance vs NumPy, localisation of injected region, determinism. Real-data run: DONE.

## 6. Actual test results
- PaDiM (real data, seed 0, paper-default random-100): 20.6 s, 45.3 MB, train peak VRAM 0.38 GB, latency 1.1 ms. Validation: image AUROC 0.874, pixel AUROC 0.971, AUPRO 0.816, Dice 0.448. Test (once): image AUROC 0.754 [0.713, 0.791], pixel AUROC 0.960 full / 0.868 brain ROI, AUPRO 0.782 / 0.476, Dice 0.419; sensitivity 0.546, specificity 0.777. Not PaDiM's best setting: on validation all-384-dim PaDiM scored image AUROC 0.912, pixel AUROC 0.976, AUPRO 0.842 (518 MB, 3.1 ms).

## 7. Bugs discovered
- -

## 8. Fixes applied
- -

## 9. Remaining issues
- Optionally evaluate PaDiM at all-384 dimensions / PCA-99 on test too (would be a further test use; not done). Report the tested PaDiM as the paper-default configuration.

## 10. Acceptance decision
- CONDITIONAL PASS (implemented and measured; single seed; tested at paper default, not its best validation setting).

## 11. Evidence and artifact references
- artifacts/models/padim_s0/*, results/logs/train_padim.txt, eval_valid_padim_s0.txt, eval_test_padim_s0.txt, docs/RESULTS.md
