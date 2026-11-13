# Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: brainsentinel` from a script | scripts add `src/` to the path themselves (tested); for ad-hoc Python use `pip install -e . --no-deps` or `PYTHONPATH=src` |
| CUDA OOM during PatchCore fit | lower `fit.max_images`, batch size (`train.batch_size`); features are kept on CPU |
| PatchCore RAM spike | 2,000 slices ≈ 2.8 GB; reduce `max_images` |
| `PermissionError ... no threshold.json` | run `evaluate.py --split valid` first |
| `PermissionError ... already evaluated` | test is single-use; `--allow-rerun` is logged |
| `train_stats` error | run `python scripts/fit_norm_stats.py` |
| `/analyze` 503 | no trained model: set `BRAINSENTINEL_MODEL_DIR` or run `select_model.py` |
| `palette_unavailable` | `python scripts/export_palette.py` |
| pretrained weights download fails | torchvision needs one-time access to download.pytorch.org; the checkpoint then embeds the weights |
| UI shows "backend offline" | start uvicorn on :8000 (Vite proxies `/api`, `/health`) |
