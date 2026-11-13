# Model design

All models: trained on `train/good` only, same preprocessing, same 240x240 grid, output (map, score) with higher = more anomalous. Implemented in plain PyTorch (not Anomalib) so every step is inspectable and independent of Anomalib's API churn; versions of torch/torchvision are the user's pins.

**A. Autoencoder** - 4 stride-2 conv blocks (width 32) to a 15x15x16 bottleneck, mirrored decoder, MSE. Early stopping on *validation normal* slices only. Map = Gaussian-smoothed |x - x̂|; score = mean of top 1% pixels (predeclared).

**B. PaDiM** - frozen ResNet18 layer2+layer3 (384-d at 30x30). Per-position Gaussian from streamed float64 moments (shifted to avoid cancellation), covariance + 0.01·I, Mahalanobis distance, bilinear upsample, Gaussian blur σ=4; score = map max. Dimension reduction: random 100 (paper), none, or PCA.

**C. PatchCore** - same backbone, 3x3 local aggregation, greedy k-center coreset (1% default) on a seeded 128-d random projection, 1-NN distance, same map post-processing; score = map max. Uses a seeded random subset of ≤2,000 train slices (RAM bound: 2.8 GB); documented deviation from "all 7,500". Paper's score re-weighting not implemented.

Deviations from the originals are listed in `TECHNICAL_DECISIONS.md`. **No model is assumed to win**; `scripts/select_model.py` applies a predeclared rule to validation metrics.
