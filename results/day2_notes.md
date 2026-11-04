# Day 2 notes (measured 2026-11-04; hypotheses are labelled)

## Representation
- 256 train-derived colours, monotone in R,G,B, within 1/255 of matplotlib `bone`.
- Exact RGB round-trip: 11,298/11,298 images. Valid/test use no colour outside the train palette.
- Index = 8-bit rendered intensity. NOT proven to be raw FLAIR. gray x3 carries the same information but is not pixel-identical to RGB: decide by ablation (Day 8).

## Index and contract (data/processed/index.csv, 13 checks PASS)
- Counts, 1:1 pairing, 240x240, mask colours (0 unknown px), empty `good` masks, no blank abnormal masks.
- Train patients disjoint from valid/test. valid∩test = 10 patients (40 valid, 50 test rows). 10 byte-identical pairs. 0 within-split duplicates.
- Lesion px: valid 10..4822 (2 under 100), test 1..6333 (67 under 100).
- Foreground colour pixel share (56,55,77)/(112,123,143)/(255,255,255): valid 0.221/0.459/0.320, test 0.149/0.494/0.357. Not interpreted clinically.

## Background offset
- 33 images (31 train: patients 01548 x22, 01551 x7, 00584 x2; 2 test/Ungood: 00446) have constant background 1..10 instead of 0.
- 8-px border equals the corner value in 11,298/11,298 images. Train-derived tolerance 10.
- Brain mask = idx > corner value (HYPOTHESIS, not a segmentation). Column `corner_bg` in the index.

## Alignment (visual, 18 overlays)
- Contours sit on hyperintense tissue. Lesion px on background: valid 0/93,756; test 184/6,731,065 (16 images, worst 1.2%).

## Distribution (descriptive; test stats must not drive decisions)
- Median brain index: train 114.6, valid/good 123.8, valid/Ungood 98.5, test/good 115.9, test/Ungood 102.6.
- Valid medians fall inside the train patient-resampling range (n=39, n=11): no detectable shift at this n.
- Ungood groups have lower gradient and mean index than good groups; good/Ungood patients are disjoint in valid and test.

## Open
- Straight vertical cut in 01432_76/84 (cause unknown). 01551 looks banded/low-res (hypothesis).
- valid/good brain fraction ~0.26 vs train ~0.28 untested.

## Carry-forward
- Use `brain_mask`/`corner_bg`, palette from train only, brain-only metrics next to full-frame, with/without sub-100 px, slice-stratified, patient-level intervals, sensitivity run excluding the 50 overlap test images.
