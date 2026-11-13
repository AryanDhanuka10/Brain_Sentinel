# Day 8 - PCA and preprocessing ablations

## 1. Original objectives
- PCA none/99/95 on embeddings, fit on train only; normalisation/augmentation ablations.

## 2. Implemented functionality
- PCAReducer (exact streamed), scripts/pca_experiments.py, predeclared keep/reject rule.

## 3. Files created or modified
- src/brainsentinel/models/reducers.py, scripts/pca_experiments.py, pipeline.pca_decision

## 4. Technical review
- PCA matches scikit-learn; never refit on later data; applied to embeddings, not pixels. Robust-percentile normalisation and augmentation ablations NOT IMPLEMENTED.

## 5. Tests executed
- test_models.py (PCA vs sklearn), test_pipeline.py (decision rule). Experiments: DONE (validation only, test untouched).

## 6. Actual test results
- Predeclared rule (each of 3 metrics drops ≤0.01 AND latency or size improves ≥20%). PatchCore: PCA-99 (250 dims) KEEP (largest drop 0.0099 image AUROC, borderline and within noise of 83 slices; model -24%, latency -16%); PCA-95 REJECT (AUPRO -0.019). PaDiM vs all-384 baseline: PCA-99 KEEP (largest drop 0.0076; size -47%, latency -28%); PCA-95 REJECT (AUPRO -0.025); random-100 REJECT (image AUROC -0.039).
- The selected/tested PatchCore model does not use PCA; PCA is reported as an optional compression with validation-only evidence.

## 7. Bugs discovered
- First run of `pca_experiments.py` crashed: `"pca" in tag` also matched `nopca`, whose IdentityReducer has no `explained` attribute.

## 8. Fixes applied
- Changed to `tag.startswith("pca")`; regression test added; results reproduced after rerun.

## 9. Remaining issues
- Robust-percentile normalisation and augmentation ablations are not implemented.

## 10. Acceptance decision
- CONDITIONAL PASS (PCA experiments complete; preprocessing ablations 2/3 missing).

## 11. Evidence and artifact references
- results/pca_experiments_patchcore.md, results/pca_experiments_padim.md, docs/PCA_EXPERIMENTS.md, results/logs/pca_*.txt
