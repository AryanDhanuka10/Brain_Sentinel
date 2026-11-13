# Known limitations

* Single seed (seed 0) for every model; no run-to-run variance estimate. Intervals are patient-bootstrap CIs over test images, not over training runs.
* Brain-ROI localisation is much weaker than full-frame numbers suggest (PatchCore AUPRO 0.583 in the brain vs 0.847 full frame; pixel AUROC 0.922 vs 0.979): the black background is easy.
* At the validation-frozen threshold PatchCore flags 12.2% of normal test slices and misses 21.1% of abnormal ones (sensitivity 0.789). Thresholds come from only 83 validation slices.
* The autoencoder baseline fails (image AUROC 0.254, specificity 0.000, ROI pixel AUROC 0.497); PaDiM was tested at its paper default, not its best validation setting.
* Slice-stratified image AUROC is 0.844 in the 60s band; the 70s/80s bands have too few normals (5 and 8) to interpret.
* Single dataset (BraTS-derived FLAIR, tumour-type abnormality); no external-site or scanner validation; not clinical.
* Valid/test patient overlap (10 patients) and slice-position confounding (see DATASET_DOCUMENTATION).
* Validation has 83 slices: thresholds are noisy; test is 82.8% abnormal.
* Patient ID / slice index are filename assumptions.
* Domain check cannot detect another sequence rendered through the same palette.
* Evidence strength is heuristic; anomaly score ≠ probability.
* Not implemented: robust-percentile and augmentation ablations, EfficientAD, IXI external test, MLflow, multi-seed runs (supported, not run), FAISS.
* Docker files unverified; Day-2 notebooks (`notebooks/*.ipynb`) and montage figures are empty/missing in the supplied repo.
* PatchCore trains on a ≤2,000-slice subset by default.
