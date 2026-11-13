"""Evaluation metrics for anomaly detection (image level) and localisation (pixel level).

Definitions (all documented in docs/EVALUATION_METHODOLOGY.md):

* Image AUROC / AUPR: ranking of per-image anomaly scores against image labels (abnormal = positive).
* Pixel AUROC / AUPR: ranking of all pixels pooled over the evaluated images against the mask.
* AUPRO: area under the per-region-overlap curve vs. pooled false-positive rate, integrated for
  FPR <= ``fpr_limit`` (default 0.3) and divided by ``fpr_limit``. Regions are 8-connected components
  of the ground-truth mask. Negatives are every non-lesion pixel of every evaluated image.
* Dice / IoU: at a *given* pixel threshold (chosen on validation, never here). Reported pooled over all
  evaluated pixels and as the mean over abnormal images (per-image), because pooled Dice is dominated
  by large lesions while per-image Dice treats a 10-px lesion like a 5000-px one.

Pixel metrics are computed from exact pixel-count histograms over *data-driven quantile bin edges*
(``PixelStats``), so 214M pixels (the full test split) never have to be held as one sorted array.
Counts per bin are exact; the only approximation is that pixels inside the same bin are treated as tied.
``tests/test_metrics.py`` bounds that error against scikit-learn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

# --------------------------------------------------------------------------------------------------
# Image level
# --------------------------------------------------------------------------------------------------


def image_auroc_aupr(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Threshold-free image metrics. ``labels``: 1 = abnormal. Needs both classes."""
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("image AUROC/AUPR need both classes")
    if not np.isfinite(s).all():
        raise ValueError("non-finite image scores")
    return {"image_auroc": float(roc_auc_score(y, s)), "image_aupr": float(average_precision_score(y, s))}


def image_threshold_metrics(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict[str, float]:
    """Confusion-matrix metrics at a *given* threshold (flag if score >= thr)."""
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(int)
    pred = s >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    return {"threshold": float(thr), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "sensitivity": rec, "specificity": div(tn, tn + fp),
            "fpr": div(fp, fp + tn), "f1": f1}


def patient_bootstrap_ci(
    scores: np.ndarray, labels: np.ndarray, patients: Sequence[str],
    fn: Callable[[np.ndarray, np.ndarray], float], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI of ``fn(scores, labels)`` resampling *patients* (not slices) with replacement.

    Slices of one patient are correlated, so resampling slices would give over-confident intervals.
    Resamples lacking one of the two classes are skipped (counted in the returned CI via the rest).
    """
    s, y, p = np.asarray(scores), np.asarray(labels), np.asarray(patients)
    ids = np.unique(p)
    rows = {i: np.flatnonzero(p == i) for i in ids}
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([rows[i] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(fn(s[idx], y[idx])))
    if len(vals) < max(10, n_boot // 10):
        raise ValueError(f"only {len(vals)} usable bootstrap resamples")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# --------------------------------------------------------------------------------------------------
# Pixel level
# --------------------------------------------------------------------------------------------------


def _chunks(n: int, size: int):
    for a in range(0, n, size):
        yield a, min(a + size, n)


def make_edges(maps: np.ndarray, n_bins: int = 4096, sample_px: int = 2_000_000, seed: int = 0,
               chunk: int = 64) -> np.ndarray:
    """Strictly increasing bin edges: exact global min/max + quantiles of a seeded pixel sample.

    Quantile edges put resolution where the scores actually are (most pixels are near the low end).
    """
    n = len(maps)
    lo, hi = np.inf, -np.inf
    for a, b in _chunks(n, chunk):
        c = np.asarray(maps[a:b])
        if not np.isfinite(c).all():
            raise ValueError("non-finite values in anomaly maps")
        lo, hi = min(lo, float(c.min())), max(hi, float(c.max()))
    rng = np.random.default_rng(seed)
    per_img = max(1, sample_px // max(n, 1))
    pool = []
    for i in range(n):
        flat = np.asarray(maps[i]).ravel()
        pool.append(flat[rng.integers(0, flat.size, size=min(per_img, flat.size))])
    q = np.quantile(np.concatenate(pool), np.linspace(0, 1, n_bins + 1)[1:-1])
    edges = np.unique(np.concatenate([[lo], q, [hi]]))
    if len(edges) < 3:  # constant map
        edges = np.array([lo - 1.0, (lo + hi) / 2, hi + 1.0])
    return edges


def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


@dataclass
class PixelStats:
    """Pooled per-bin pixel counts of lesion (pos) and non-lesion (neg) pixels + per-region histograms."""

    edges: np.ndarray                 # (K+1,) increasing; bin j = [edges[j], edges[j+1])
    pos: np.ndarray                   # (K,) int64
    neg: np.ndarray                   # (K,) int64
    region_hist: np.ndarray           # (R, K) int32, one row per connected GT component
    region_size: np.ndarray           # (R,) int64
    n_images: int

    @classmethod
    def build(cls, maps: np.ndarray, masks: np.ndarray, roi: np.ndarray | None = None,
              edges: np.ndarray | None = None, n_bins: int = 4096, seed: int = 0, chunk: int = 64,
              ) -> "PixelStats":
        """maps (N,H,W) float, masks (N,H,W) {0,1}, roi (N,H,W) bool or None (None = full frame)."""
        if maps.shape != masks.shape:
            raise ValueError(f"maps {maps.shape} != masks {masks.shape}")
        if roi is not None and roi.shape != masks.shape:
            raise ValueError("roi shape mismatch")
        edges = make_edges(maps, n_bins=n_bins, seed=seed, chunk=chunk) if edges is None else np.asarray(edges)
        K = len(edges) - 1
        pos = np.zeros(K, np.int64)
        neg = np.zeros(K, np.int64)
        rh: list[np.ndarray] = []
        rs: list[int] = []
        s8 = np.ones((3, 3), dtype=int)
        for a, b in _chunks(len(maps), chunk):
            m = np.asarray(masks[a:b]) != 0
            sc = np.asarray(maps[a:b])
            keep = np.ones_like(m) if roi is None else np.asarray(roi[a:b]).astype(bool)
            bi = _bin_index(sc, edges)
            pos += np.bincount(bi[m & keep], minlength=K)
            neg += np.bincount(bi[~m & keep], minlength=K)
            for j in range(b - a):
                if not m[j].any():
                    continue
                lab, nlab = ndimage.label(m[j], structure=s8)
                for r in range(1, nlab + 1):
                    sel = (lab == r) & keep[j]
                    if sel.any():
                        rh.append(np.bincount(bi[j][sel], minlength=K).astype(np.int32))
                        rs.append(int(sel.sum()))
        region_hist = np.stack(rh) if rh else np.zeros((0, K), np.int32)
        return cls(edges, pos, neg, region_hist, np.asarray(rs, np.int64), len(maps))

    # -- curves over thresholds edges[j] (flag pixel if score >= edges[j]) ----------------------
    def _above(self) -> tuple[np.ndarray, np.ndarray]:
        """tp[j], fp[j] = pixels with score >= edges[j]."""
        return np.cumsum(self.pos[::-1])[::-1], np.cumsum(self.neg[::-1])[::-1]

    @property
    def n_pos(self) -> int:
        return int(self.pos.sum())

    @property
    def n_neg(self) -> int:
        return int(self.neg.sum())

    def pixel_auroc(self) -> float:
        if self.n_pos == 0 or self.n_neg == 0:
            raise ValueError("pixel AUROC needs lesion and non-lesion pixels")
        tp, fp = self._above()
        tpr = np.concatenate([[0.0], tp[::-1] / self.n_pos])      # thresholds from high to low
        fpr = np.concatenate([[0.0], fp[::-1] / self.n_neg])
        return float(np.trapezoid(tpr, fpr))

    def pixel_aupr(self) -> float:
        """Average precision (step-wise), same definition as scikit-learn."""
        if self.n_pos == 0:
            raise ValueError("pixel AUPR needs lesion pixels")
        tp, fp = self._above()
        tp, fp = tp[::-1].astype(np.float64), fp[::-1].astype(np.float64)   # high -> low threshold
        denom = tp + fp
        prec = np.divide(tp, denom, out=np.ones_like(tp), where=denom > 0)
        rec = tp / self.n_pos
        return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))

    def aupro(self, fpr_limit: float = 0.3) -> float:
        """Area under mean per-region overlap vs pooled FPR on [0, fpr_limit], divided by fpr_limit."""
        if not 0 < fpr_limit <= 1:
            raise ValueError("fpr_limit must be in (0, 1]")
        if len(self.region_size) == 0 or self.n_neg == 0:
            raise ValueError("AUPRO needs at least one GT region and non-lesion pixels")
        above = np.cumsum(self.region_hist[:, ::-1], axis=1)             # (R,K) pixels >= threshold, high->low
        pro = (above / self.region_size[:, None]).mean(axis=0)
        _, fp = self._above()
        fpr = fp[::-1] / self.n_neg
        fpr = np.concatenate([[0.0], fpr])
        pro = np.concatenate([[0.0], pro])
        # fpr is non-decreasing; integrate up to the limit with a linear cut at the boundary
        m = fpr <= fpr_limit
        x, y = fpr[m], pro[m]
        if x[-1] < fpr_limit and m.sum() < len(fpr):
            j = int(m.sum())
            x1, x2, y1, y2 = fpr[j - 1], fpr[j], pro[j - 1], pro[j]
            yl = y1 + (y2 - y1) * (fpr_limit - x1) / (x2 - x1) if x2 > x1 else y1
            x, y = np.append(x, fpr_limit), np.append(y, yl)
        return float(np.trapezoid(y, x) / fpr_limit)

    def dice_iou_curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(thresholds, pooled dice, pooled iou) for every bin edge (edges[:-1])."""
        tp, fp = self._above()
        fn = self.n_pos - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            dice = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
            iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
        return self.edges[:-1], dice, iou


def dice_iou_at(maps: np.ndarray, masks: np.ndarray, thr: float, roi: np.ndarray | None = None,
                chunk: int = 64) -> dict[str, float]:
    """Exact Dice/IoU at a given pixel threshold: pooled, and mean per abnormal image.

    ``normal_fp_px_rate`` = flagged pixels / pixels on images with an empty mask (how noisy normals are).
    """
    tp = fp = fn = 0
    per_dice: list[float] = []
    per_iou: list[float] = []
    nfp = npx = 0
    for a, b in _chunks(len(maps), chunk):
        m = np.asarray(masks[a:b]) != 0
        p = np.asarray(maps[a:b]) >= thr
        if roi is not None:
            k = np.asarray(roi[a:b]).astype(bool)
            m, p = m & k, p & k
        for j in range(b - a):
            t = int((p[j] & m[j]).sum())
            f = int((p[j] & ~m[j]).sum())
            n = int((~p[j] & m[j]).sum())
            tp, fp, fn = tp + t, fp + f, fn + n
            if m[j].any():
                per_dice.append(2 * t / (2 * t + f + n))
                per_iou.append(t / (t + f + n))
            else:
                nfp += int(p[j].sum())
                npx += p[j].size
    div = lambda x, y: float(x / y) if y else float("nan")  # noqa: E731
    return {"threshold": float(thr),
            "dice_pooled": div(2 * tp, 2 * tp + fp + fn), "iou_pooled": div(tp, tp + fp + fn),
            "dice_per_image": float(np.mean(per_dice)) if per_dice else float("nan"),
            "iou_per_image": float(np.mean(per_iou)) if per_iou else float("nan"),
            "normal_fp_px_rate": div(nfp, npx)}
