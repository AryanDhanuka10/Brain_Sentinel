# Day 3 notes (preprocessing + dataset; hypotheses are labelled; TBD = not yet measured)

## Decisions (fixed before any model result)
- Native 240x240, no resize. Masks are never resized; later score maps are upsampled to 240 for scoring.
- Input = recovered index/255, gray replicated x3, fixed ImageNet mean/std (no fitting). `train_stats` mode
  (scalar mean/std over train/good brain pixels) exists for the autoencoder/ablations only.
- Per-image background: subtract own `corner_bg`, clip at 0. `brain_mask` (HYPOTHESIS) is returned as metadata,
  never used to zero pixels. RGB tint = Day 8 ablation.
- No augmentation code exists. No histogram equalisation. Valid/test use the identical deterministic path.

## Guards (in code, each with a regression test)
- Dataset refuses any split whose index rows differ from its files on disk, or whose size differs from `expected_n`.
- Refuses any train patient that also appears in valid/test (documented valid/test overlap is allowed;
  `exclude_patients` drops the 50 overlap test images for the sensitivity run).
- `corner_bg` must be <= 10 (train-derived) and must equal index.csv; unseen colours, wrong size, non-finite
  output and corrupt PNGs raise. Train palette must be exactly 256 monotone colours.

## Measured
- pytest: 82 passed, 1 skipped (norm-stats test, before fit_norm_stats ran). After fit: TBD.
- Real-file test facts that passed: lengths 7500/83/3715; sensitivity exclusion leaves 3665; valid lesions
  0 px outside brain; 67 test lesions < 100 px keep their pixels; masks binary and equal to index lesion_px
  (all valid, every 37th test item); 01548_77 background 6 -> 0.
- norm_stats.json (train/good brain pixels): mean TBD, std TBD, n_pixels TBD, file_list_sha256 TBD.
- Full-corpus audit (scripts/audit_day3.py, results/day3_audit.json): hard problems TBD;
  Day 2 reproduction valid TBD / test TBD; brain fraction train TBD, valid/good TBD (Day 2 open item, untested).

## Open / not verified
- Before/after montages through the pipeline: NOT RUN (alignment evidence is still the Day 2 raw overlays).
- Vertical cut in 01432_76/84 and banding in 01551: still unexplained.
- Image-level labels remain confounded with patient/volume (Day 2 finding 7).

## Carry-forward
- Use `brain`/`corner_bg` metadata for brain-only metrics next to full-frame, slice-stratified, with/without
  sub-100 px, patient-level intervals, and the sensitivity run via `exclude_patients`.
- Thresholds/hyperparameters on validation only; test untouched until Day 9.
