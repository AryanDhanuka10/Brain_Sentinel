# Data preprocessing

Fixed in code (Day 3, unchanged here): native 240x240 (no resize), recover the palette index (bijective, train-derived palette), subtract each image's own corner background (0 for most; 1-10 for 33 known train images), `index/255`, replicate to 3 channels, normalise. Normalisation modes (`configs/preprocess.yaml`): `imagenet` (fixed constants; used by PaDiM/PatchCore) or `train_stats` (scalar mean 0.45215 / std 0.18039 fitted on 7,500 train/good brain pixels, frozen; used by the autoencoder). No histogram equalisation, no augmentation by default.

Masks: foreground = RGB != 0, never resized. Corrupt/unknown-colour/odd-corner images raise; nothing is silently skipped.

**Serving uses the same code path** (`inference.checks.decode_upload` mirrors `load_index_image`; a test asserts bit-identical output). The run's resolved normalisation constants are stored in `config.json`, so serving never depends on `data/processed/`.

Not implemented (declared, not hidden): robust-percentile normalisation ablation, conservative training augmentation, RGB-tint ablation. Status NOT RUN.
