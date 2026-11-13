# Final acceptance report

**Overall decision: CONDITIONALLY ACCEPTED - code-complete and measured on real BMAD data (single seed); submission deliverables (slides, demo recording, multi-seed, error-analysis plots) are still owed.**

| Area | Decision | Basis |
|---|---|---|
| Days 0-3 retrospective | PASS / CONDITIONAL PASS | user-run evidence reviewed; 6 defects fixed with regression tests; notebook/montage still missing |
| Evaluation framework | PASS | 17 metric tests vs independent references; protocol guards (frozen thresholds, test once) verified by `evaluations.jsonl` |
| Models A/B/C | measured | `docs/RESULTS.md`; PatchCore selected on validation only |
| PCA | measured (validation only) | PatchCore PCA-99 and PaDiM PCA-99 pass the predeclared rule (PatchCore borderline); selected model uses no PCA |
| Agent workflow | CONDITIONAL PASS | 30 tests; one real-model run seen by the user |
| API + UI | CONDITIONAL PASS | backend + frontend ran with the selected model; 503/413/500, uncertain and quality-failed states not run on the real model; Docker unverified |
| Slides / demo / resume numbers | OPEN | numbers are ready in `docs/RESULTS.md` |

Test suite on the author's machine: **212 passed, 0 failed** (`results/logs/pytest_full.txt`).

Headline (test, evaluated once): PatchCore image AUROC 0.909 [0.884, 0.931], pixel AUROC 0.979 full frame / 0.922 brain ROI, AUPRO 0.847 / 0.583, Dice 0.501; sensitivity 0.789 and specificity 0.878 at the validation-frozen threshold. The autoencoder baseline fails (image AUROC 0.254). Research flags for qualified human review, not diagnosis.

Honest caveats: single seed; brain-ROI localisation is clearly weaker than full-frame numbers; 83-image validation set; PaDiM tested at its paper default; slice-stratified AUROC 0.844 in the 60s band.

Remaining before submission: (1) slides from `docs/PRESENTATION_OUTLINE.md` + `docs/RESULTS.md` only; (2) demo recording/screenshots (clear, flagged, and the normal-slice false positive); (3) optional multi-seed and error-analysis panels; (4) `pip freeze > requirements-lock.txt`; tag; submit.

Audit-trail notes: `results/experiment_log.csv` keeps two defects visible - a FAILED autoencoder row (seed bug, fixed) and a `patchcore_s0` row marked OK with no runtime (run interrupted; logging bug fixed so interrupts now log INTERRUPTED; the complete run follows).
