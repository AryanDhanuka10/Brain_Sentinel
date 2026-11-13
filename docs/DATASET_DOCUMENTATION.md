# Dataset documentation

Primary dataset: **BMAD Brain** (Bao et al., arXiv:2306.11876), derived from **BraTS 2021 FLAIR**; 240x240 RGBA PNG axial slices. Source, licence (CC BY-NC-SA), citations, download date, manifest hash: `data/README.md`. The raw BraTS NIfTI volumes mentioned in the project brief are *not* used; the audit established that the downloaded data is the preprocessed BMAD benchmark (`Brain_AD.zip`).

Official splits (measured locally by the user, unmodified; match published 7,500 / 83 / 3,715):

| Split | Class | Images | Patients (filename prefix) |
|---|---|---|---|
| train | good | 7,500 | 424 |
| valid | good | 39 | 39 |
| valid | Ungood | 44 | 11 |
| test | good | 640 | 162 |
| test | Ungood | 3,075 | 615 |

Findings that shape the experiments (all from the user's audit outputs; see `results/day2/`):
* 10 patients appear in both valid/Ungood and test/Ungood (10 byte-identical image+mask pairs; 50 test images affected). Report official test metrics AND a patient-disjoint run (`patient_disjoint` block in `metrics_test.json`).
* Slice position is confounded with the label (test normals are 98% in the 60s/90s bins; 70-89 is ~99% abnormal): report slice-stratified image AUROC (`slice_stratified_image`).
* Images are palette-rendered (one 8-bit intensity through a fixed 256-colour map), not true grayscale. Masks are RGB with 4 allowed colours; foreground = RGB != 0; nearest-neighbour only.
* Patient ID and slice index are *assumed* from filenames (not documented by the source).
* "Normal" = BMAD normal-training slices from BraTS cases, not healthy volunteers.
