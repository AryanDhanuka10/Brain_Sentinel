# Day 3 - Preprocessing and tests (retrospective)

## 1. Original objectives
- Deterministic preprocessing, norm stats fitted on train only, tests, before/after montage.

## 2. Implemented functionality
- Palette recovery, corner-background subtraction, index/255, replicate x3, imagenet or train_stats normalisation, BMADBrain dataset, norm_stats.json (mean 0.45215, std 0.18039, 123,420,394 px).

## 3. Files created or modified
- results/day3/day3_notes.md (modified); src/brainsentinel/data/* reviewed, unchanged

## 4. Technical review
- Code reviewed line by line: no test-fitted statistics (fitted_on == train/good enforced), nearest-neighbour masks, no augmentation/equalisation, deterministic transforms. Stale note fixed: day3_notes.md still said 'mean TBD'. The serving path was verified to be bit-identical to the training loader (new test). Missing: before/after montage; user's post-fit pytest output not supplied.

## 5. Tests executed
- User-run suite and logs under results/ (not re-executed here: needs GPU/dataset); in the sandbox only the tests named under 'Actual test results' were run.

## 6. Actual test results
- Sandbox: new test_decode_roundtrip_matches_training_loader passes; the 71 data-gated preprocessing tests skip.

## 7. Bugs discovered
- Stale TBD in notes.

## 8. Fixes applied
- Updated with the user-supplied measured values.

## 9. Remaining issues
- Paste local `pytest -q` after fit_norm_stats.py; produce before/after montage.

## 10. Acceptance decision
- CONDITIONAL PASS

## 11. Evidence and artifact references
- data/processed/norm_stats.json (user-pasted)
- results/day3/*

