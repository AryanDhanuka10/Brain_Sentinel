# Interview preparation (answers must be updated with your measured results)

* **Why unsupervised?** Few pixel labels, abnormality is open-ended; normal data is abundant. Caveat: validation still uses labels (normal-only training with supervised evaluation).
* **Why three models?** Different paradigms: reconstruction vs Gaussian feature statistics vs memory bank. State the measured winner and the predeclared rule - do not pre-assume PatchCore.
* **How did you prevent leakage?** Official patient-separated splits; train-only palette/norm stats/PCA; thresholds frozen from valid (`threshold.json` refuses other splits); test gated and logged. Honest flaw: 10 patients overlap valid/test → patient-disjoint run.
* **Why AUPRO / per-image Dice?** Pooled Dice is dominated by large lesions; AUPRO weights each region equally.
* **What is slice confounding?** Test normals sit in the 60s/90s slice bins; image AUROC can partly reward slice position → stratified AUROC.
* **PCA?** Applied to embeddings, train-only, kept only if ≤0.01 metric loss and ≥20% speed/size gain; say what happened.
* **What makes it agentic?** Tool-calling nodes, state-dependent routing, retry, escalation, human gate; LLM optional and guarded.
* **Is it a diagnosis?** No; anomaly ≠ disease; not clinically validated; domain check = format conformance only.
* **Weak spots to own:** 83-image validation set; heuristic evidence rules; PatchCore on a 2,000-slice subset; single dataset.
