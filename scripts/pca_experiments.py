"""PCA ablation (validation only; the test split is never touched).

    python scripts/pca_experiments.py --model patchcore          # none vs PCA 99 % vs PCA 95 %
    python scripts/pca_experiments.py --model padim              # none(384) vs random-100 (paper) vs PCA 99/95
PCA is fitted on train/good embeddings only (see brainsentinel.models.reducers). The keep/reject decision
uses the predeclared rule in brainsentinel.pipeline.pca_decision and is written to results/pca_experiments.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import evaluate as evaluate_cli  # noqa: E402
import train as train_cli  # noqa: E402
from brainsentinel.pipeline import pca_decision  # noqa: E402

VARIANTS = {
    "patchcore": {"nopca": ["hparams.reduction=none"],
                  "pca99": ["hparams.reduction=pca", "hparams.pca_variance=0.99"],
                  "pca95": ["hparams.reduction=pca", "hparams.pca_variance=0.95"]},
    "padim": {"nopca384": ["hparams.reduction=none"],
              "rand100": ["hparams.reduction=random", "hparams.d_out=100"],
              "pca99": ["hparams.reduction=pca", "hparams.pca_variance=0.99"],
              "pca95": ["hparams.reduction=pca", "hparams.pca_variance=0.95"]},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rows = {}
    for tag, sets in VARIANTS[a.model].items():
        rid = f"{a.model}_{tag}_s{a.seed}"
        run = REPO / "artifacts" / "models" / rid
        if not (run / "model.pt").exists():
            train_cli.main(["--model", a.model, "--tag", tag, "--seed", str(a.seed), "--device", a.device, "--set", *sets])
        if not (run / "metrics_valid.json").exists():
            evaluate_cli.main(["--run", str(run), "--split", "valid", "--device", a.device, "--n-boot", "200"])
        m = json.loads((run / "metrics_valid.json").read_text())
        c = json.loads((run / "config.json").read_text())
        pf, im, eng = m["official"]["pixel_full_frame"], m["official"]["image"], m["engineering"]
        rows[tag] = {"tag": tag, "d_used": c["fit_info"].get("d_used"), "image_auroc": im["image_auroc"],
                     "pixel_auroc": pf["pixel_auroc"], "aupro": pf["aupro_fpr0.3"], "dice": pf.get("dice_per_image"),
                     "latency_ms": eng["latency_ms_mean"], "peak_vram_gb": eng.get("peak_vram_gb"),
                     "model_mb": c["checkpoint_mb"], "fit_s": c["fit_info"].get("fit_seconds"),
                     "explained": None}
        if tag.startswith("pca"):          # NB: "nopca" also contains "pca" but has an IdentityReducer
            from brainsentinel.models import load_model
            rows[tag]["explained"] = float(load_model(run).reducer.explained)
    base = rows["nopca"] if "nopca" in rows else rows["nopca384"]
    lines = [f"# PCA experiments: {a.model} (validation only)", "",
             "| Variant | dims | explained var. | Image AUROC | Pixel AUROC | AUPRO | Dice | Latency ms | Peak VRAM GB | Model MB | Fit s | Decision vs no-PCA |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, n=4: "n/a" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))  # noqa: E731
    for tag, r in rows.items():
        dec = "baseline" if r is base else ("KEEP" if pca_decision(base, r)["keep"] else "REJECT")
        lines.append(f"| {tag} | {r['d_used']} | {f(r['explained'])} | {f(r['image_auroc'])} | {f(r['pixel_auroc'])} | "
                     f"{f(r['aupro'])} | {f(r['dice'])} | {f(r['latency_ms'], 1)} | {f(r['peak_vram_gb'], 2)} | "
                     f"{f(r['model_mb'], 1)} | {f(r['fit_s'], 1)} | {dec} |")
    lines += ["", "Rule: keep only if every quality drop (image AUROC, pixel AUROC, AUPRO) <= 0.01 AND latency or model size improves >= 20 %.", ""]
    (REPO / "results" / f"pca_experiments_{a.model}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
