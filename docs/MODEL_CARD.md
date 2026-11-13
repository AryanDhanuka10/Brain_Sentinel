# Model card: BrainSentinel selected model (PatchCore, `patchcore_s0`)

**Intended use:** research prototype that flags brain-MRI FLAIR axial slices (BMAD Brain format) and localises anomalous regions for *qualified human review*. **Not a diagnostic device**; scores are not disease probabilities; never use for clinical decisions.

**Model:** ResNet18 (ImageNet) layer2+layer3 patch features (384-d, 3x3 aggregation), greedy k-center coreset memory bank (1% of 1.8 M patches = 18,000), 1-NN distance, sigma-4 Gaussian blur. Trained only on normal slices (2,000 random `train/good` slices, seed 0). 37.0 MB checkpoint, no PCA.

**Data:** BMAD Brain (BraTS 2021 FLAIR, 240x240 PNG). Train/good 7,500; valid 83 (39 normal, 44 abnormal); test 3,715 (640 normal, 3,075 abnormal). Thresholds chosen on validation only (image max-F1 = 2.2374; pixel max-Dice = 2.1663); test evaluated once.

**Measured performance (test, single seed):** image AUROC 0.909 (95% patient-bootstrap CI 0.884-0.931); sensitivity 0.789, specificity 0.878 (12.2% of normal slices flagged); pixel AUROC 0.979 full frame / 0.922 brain ROI; AUPRO 0.847 / 0.583; Dice per-image 0.501 (pooled 0.600). Patient-disjoint run: image AUROC 0.909. Slice-stratified image AUROC: 0.844 (60s band), 0.955 (90s band). Latency 4.5 ms/slice (GPU, batch 1), inference VRAM 0.13 GB. Full tables: `docs/RESULTS.md`.

**Limitations:** single dataset and scanner source, tumour-type abnormalities only; single seed; weaker localisation inside the brain than full-frame metrics imply; small validation set (83) makes thresholds noisy; the domain check verifies file format only, not modality; evidence-strength is a heuristic review signal; 10 patients overlap valid and test in the benchmark (patient-disjoint run reported). Not validated on other sequences, sites, or populations.

**Ethics:** human review is the final step of the workflow; reports use review-flag language only. See `docs/KNOWN_LIMITATIONS.md`.
