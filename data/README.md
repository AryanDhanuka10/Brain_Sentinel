# Data: BMAD Brain subset (BraTS 2021 FLAIR, 2D axial PNG)
Not in Git. Raw data lives in `data/raw/Brain/`.

## Source
- Benchmark: Bao, Sun, Deng, He, Zhang, Li. "BMAD: Benchmarks for Medical Anomaly Detection", arXiv:2306.11876.
  Repo: https://github.com/DorisBao/BMAD
- Original data: BraTS 2021 (challenge data may be used provided the challenge citations are given).
- BraTS citations (exact entries from the challenge page): [Baid 2021](https://arxiv.org/abs/2107.02314),[Menze 2015](https://pubmed.ncbi.nlm.nih.gov/25494501/),[Bakas 2017](https://pubmed.ncbi.nlm.nih.gov/28872634/)
- Downloaded from : [Google Drive URL](https://drive.google.com/drive/folders/1AC-wWZl_K18CWL2eIxUScoSOoxT4IBuw?usp=sharing)
- Folder/file name downloaded: Brain_AD.zip | Download date: 03.11.2026
- Integrity manifest: data/MANIFEST.sha256 (SHA-256 of the manifest: 76d22b675cdc2d442e5a73bb4851244b47d71a0aaaff229078f0917aef908c9e)

## License
BMAD repository states CC BY-NC-SA: non-commercial use, attribution, share-alike.

## Official splits (measured locally, unmodified; match published totals 7,500 / 83 / 3,715)
| Split | Class | Images | Patients (filename prefix) |
|---|---|---|---|
| train | good | 7,500 | 424 |
| valid | good | 39 | 39 |
| valid | Ungood | 44 | 11 |
| test | good | 640 | 162 |
| test | Ungood | 3,075 | 615 |

## Layout quirks vs. the BMAD README
Masks are in `label/` (README: `anomaly_mask/`). `train/good` is flat (no `img/`). `good/` has an all-empty `label/`.

## Encoding
240x240 RGBA uint8, alpha=255, R/G/B not identical (not grayscale); colour structure in results/day1_notes.md.
Masks: RGB in {(0,0,0),(56,55,77),(112,123,143),(255,255,255)}. Foreground = RGB != 0. Nearest-neighbour only.

## Known issues and limitations
- 10 byte-identical image+mask pairs are in both valid/Ungood and test/Ungood (slice 60 of patients
  00124 00134 00154 00191 00630 01180 01388 01471 01597 01646).
- Those 10 patients also have different slices in each split (valid 60/70/80/90, test 60/68/76/84/92):
  40 of 44 valid abnormal images (10 of 11 valid/Ungood patients) share patients with test; 50 test images are affected.
- Train is disjoint from valid and test. Official splits are unmodified. Thresholds chosen on valid may be slightly
  optimistic for test. Report official test metrics AND a patient-disjoint sensitivity run (50 test images excluded).
- Validation has only 83 slices, so threshold estimates are noisy.
- Test is 82.8% abnormal; lesions range down to about 1 pixel.
- Patient ID = filename prefix (assumption, not documented by the source); slice index = filename suffix (same caveat).
- Research use only. No clinical claims.
EOF