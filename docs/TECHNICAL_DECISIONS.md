# Technical decisions

1. **BMAD PNGs, not raw BraTS NIfTI**: the brief mentions NIfTI; the real download is BMAD (plan §1/§5). No NiBabel pipeline.
2. **Own PaDiM/PatchCore instead of Anomalib**: transparent, no extra dependency, no API drift; costs: no FAISS, no re-weighting. Both documented.
3. **PatchCore uses ≤2,000 random train slices** (host RAM); coreset 1%. Ablate `max_images` locally if RAM allows.
4. **bilinear embedding concat** in PaDiM (original unfolds).
5. **Binned exact-count pixel metrics** (memory); error bounded by tests.
6. **Thresholds: max-F1 (image), max pooled Dice (pixel)** on valid; sensitivity-target rule exists (`sens_at_least`).
7. **Test guard in code** (frozen thresholds; run once; logged reruns) - audit trail, not tamper-proof.
8. **Upload domain = format conformance**; non-palette images rejected.
9. **LLM optional and guarded**; deterministic report is the product.
10. **Retry once** on inference failure; degrade to a report, never crash.
11. **Heatmap colour scale fixed to the validation pixel threshold** so slices are comparable.
12. SQLite for jobs/reports/reviews; images on disk.
