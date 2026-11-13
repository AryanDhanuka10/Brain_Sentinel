# PCA experiments: padim (validation only)

| Variant | dims | explained var. | Image AUROC | Pixel AUROC | AUPRO | Dice | Latency ms | Peak VRAM GB | Model MB | Fit s | Decision vs no-PCA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nopca384 | 384 | n/a | 0.9120 | 0.9758 | 0.8420 | 0.4409 | 3.1 | 0.52 | 518.2 | 24.1 | baseline |
| rand100 | 100 | n/a | 0.8735 | 0.9714 | 0.8156 | 0.4479 | 1.1 | 0.05 | 45.3 | 20.3 | REJECT |
| pca99 | 278 | 0.9902 | 0.9044 | 0.9741 | 0.8345 | 0.4370 | 2.3 | 0.28 | 277.4 | 19.6 | KEEP |
| pca95 | 158 | 0.9504 | 0.9021 | 0.9709 | 0.8168 | 0.4418 | 1.5 | 0.11 | 97.1 | 19.1 | REJECT |

Rule: keep only if every quality drop (image AUROC, pixel AUROC, AUPRO) <= 0.01 AND latency or model size improves >= 20 %.

