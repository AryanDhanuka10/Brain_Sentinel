# PCA experiments: patchcore (validation only)

| Variant | dims | explained var. | Image AUROC | Pixel AUROC | AUPRO | Dice | Latency ms | Peak VRAM GB | Model MB | Fit s | Decision vs no-PCA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nopca | 384 | n/a | 0.9301 | 0.9855 | 0.8978 | 0.5195 | 4.5 | 0.13 | 37.0 | 370.4 | baseline |
| pca99 | 250 | 0.9900 | 0.9202 | 0.9867 | 0.8904 | 0.5502 | 3.8 | 0.11 | 28.2 | 368.1 | KEEP |
| pca95 | 116 | 0.9502 | 0.9167 | 0.9863 | 0.8791 | 0.5450 | 3.1 | 0.10 | 18.8 | 333.3 | REJECT |

Rule: keep only if every quality drop (image AUROC, pixel AUROC, AUPRO) <= 0.01 AND latency or model size improves >= 20 %.

