# Model comparison (measured results only)

Source: `artifacts/models/<run>/metrics_<split>.json`. A cell reads NOT RUN when no such file/value exists.
Pixel metrics are full-frame; Dice/IoU are the mean over abnormal images at the validation-frozen pixel threshold;
FPR is the image-level false-positive rate at the validation-frozen image threshold. Latency is per slice (batch 1).

## valid

| Run | Model | Image AUROC | Image AUPR | Pixel AUROC | AUPRO | Dice | IoU | FPR | Latency (ms) | Peak VRAM (GB) |
|---|---|---|---|---|---|---|---|---|---|---|
| autoencoder_s0 | autoencoder | 0.3520 | 0.4393 | 0.8317 | 0.5015 | 0.1928 | 0.1111 | 1.0000 | 0.7 | 0.01 |
| padim_s0 | padim | 0.8735 | 0.8928 | 0.9714 | 0.8156 | 0.4479 | 0.3229 | 0.1538 | 1.1 | 0.05 |
| patchcore_s0 | patchcore | 0.9301 | 0.9455 | 0.9855 | 0.8978 | 0.5195 | 0.3956 | 0.0769 | 4.5 | 0.13 |

## test

| Run | Model | Image AUROC | Image AUPR | Pixel AUROC | AUPRO | Dice | IoU | FPR | Latency (ms) | Peak VRAM (GB) |
|---|---|---|---|---|---|---|---|---|---|---|
| autoencoder_s0 | autoencoder | 0.2540 | 0.7209 | 0.8542 | 0.5921 | 0.2071 | 0.1192 | 1.0000 | 0.7 | 0.01 |
| padim_s0 | padim | 0.7537 | 0.9195 | 0.9605 | 0.7823 | 0.4195 | 0.2994 | 0.2234 | 1.1 | 0.05 |
| patchcore_s0 | patchcore | 0.9093 | 0.9767 | 0.9789 | 0.8467 | 0.5014 | 0.3805 | 0.1219 | 4.5 | 0.13 |

All outputs are research flags for qualified human review, not diagnosis.
