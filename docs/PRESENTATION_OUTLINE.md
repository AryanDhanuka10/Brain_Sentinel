# Presentation outline (5 min). Every number must be copied from `docs/RESULTS.md` / `results/` (all measured, single seed).

1. Problem: labelled lesion masks are scarce; normal slices are plentiful. Unsupervised localisation + human review.
2. Data: BMAD Brain (BraTS 2021 FLAIR), normal-only train 7,500 / valid 83 / test 3,715; licence + citations slide.
3. Method: three paradigms (reconstruction, Gaussian features, memory bank) under one protocol.
4. Leakage controls: official splits, train-only fits (norm stats, PCA, palette), thresholds frozen on valid, test once; the 10-patient valid/test overlap and the patient-disjoint run.
5. Results: copy from `docs/RESULTS.md` - test table (PatchCore image AUROC 0.909 [0.884, 0.931], pixel AUROC 0.979 full frame / 0.922 brain ROI, AUPRO 0.847 / 0.583, Dice 0.501; PaDiM 0.754 / 0.960 / 0.868; autoencoder fails); selected model + rule; PCA decision (PatchCore PCA-99 keep, borderline: -24% size, -16% latency; PCA-95 reject). Show the brain-ROI numbers next to full-frame ones, and the 12.2% normal-slice flag rate.
6. Agentic workflow: diagram from `docs/AGENTIC_WORKFLOW.md`; show a flagged, an uncertain and a rejected example from the UI.
7. Demo (60-90 s): upload → heatmap/overlay → evidence → trace → human review.
8. Limitations and responsible use: not diagnostic, single dataset, slice-position confounding, heuristic evidence.
