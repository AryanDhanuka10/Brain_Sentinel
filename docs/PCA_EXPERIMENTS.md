# PCA experiments

**Status: MEASURED (validation only; test never touched).** Raw tables: `results/pca_experiments_patchcore.md`, `results/pca_experiments_padim.md`.

| Model | Variant | dims | Largest quality drop vs no-PCA | Size | Latency | Decision |
|---|---|---|---|---|---|---|
| PatchCore | PCA 99% | 250 | 0.0099 (image AUROC) | -24% | -16% | KEEP (borderline) |
| PatchCore | PCA 95% | 116 | 0.0187 (AUPRO) | -49% | -31% | REJECT |
| PaDiM (vs all-384) | PCA 99% | 278 | 0.0076 (image AUROC) | -47% | -28% | KEEP |
| PaDiM (vs all-384) | PCA 95% | 158 | 0.0252 (AUPRO) | -81% | -53% | REJECT |
| PaDiM (vs all-384) | random-100 | 100 | 0.0385 (image AUROC) | -91% | -64% | REJECT |

The PatchCore PCA-99 pass is within noise on 83 validation slices (drop 0.0099 vs limit 0.01): claim a smaller/faster model without measurable loss, not an improvement. The selected and tested PatchCore model does not use PCA. The PaDiM baseline is the full 384-dim model (518 MB), so the paper-default random-100 variant that was tested on the test split is rejected relative to it.

What is compared (validation only): PatchCore none / PCA 99% / PCA 95%; PaDiM none(384) / random-100 (paper) / PCA 99% / PCA 95%. PCA acts on backbone *embeddings* (never raw pixels), is fitted on train/good embeddings only (≤500 train images, streamed exact covariance; test asserts it matches scikit-learn and never refits), and is reused unchanged downstream. Reported: dims, explained variance, fit time, latency, VRAM, model size, image/pixel AUROC, AUPRO, Dice.

Predeclared decision (`pipeline.pca_decision`): **keep** only if image AUROC, pixel AUROC and AUPRO each drop ≤ 0.01 *and* latency or model size improves ≥ 20%; otherwise **reject**. Result and rationale go to `results/pca_experiments_<model>.md`. PCA is not applied to the autoencoder.
