"""Evidence verification: stability under small perturbations, location sanity, flagged regions.

All numbers come from the model's own maps. The rules that turn them into ``evidence strength`` are
explicit constants (EVIDENCE_RULES) and a heuristic review signal, NOT a calibrated clinical confidence.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

EVIDENCE_RULES = {"min_mean_map_corr": 0.85, "max_score_cv": 0.15, "max_peak_shift_px": 10,
                  "max_outside_brain_frac": 0.5, "max_border_frac": 0.5, "strong_margin_ratio": 0.25,
                  "top_frac": 0.01, "border_px": 10, "shift_px": 2, "noise_sigma_idx": 2.0,
                  "scale_factors": (0.97, 1.03), "noise_seed": 0}


def perturbations(idx: np.ndarray) -> list[tuple[str, np.ndarray, tuple[int, int]]]:
    """Deterministic, clinically conservative variants: (name, image, (dy, dx) applied)."""
    R = EVIDENCE_RULES
    out = []
    for f in R["scale_factors"]:
        out.append((f"intensity_x{f}", np.clip(idx.astype(np.float32) * f, 0, 255).round().astype(np.uint8), (0, 0)))
    s = R["shift_px"]
    for dy, dx in ((0, s), (0, -s), (s, 0), (-s, 0)):
        out.append((f"shift_dy{dy:+d}_dx{dx:+d}", np.roll(idx, (dy, dx), axis=(0, 1)), (dy, dx)))
    rng = np.random.default_rng(R["noise_seed"])
    noisy = np.clip(idx.astype(np.float32) + rng.normal(0, R["noise_sigma_idx"], idx.shape), 0, 255).round().astype(np.uint8)
    out.append((f"noise_sigma{R['noise_sigma_idx']}", noisy, (0, 0)))
    return out


def stability(base_map: np.ndarray, base_score: float, variants: list[tuple[str, np.ndarray, float, tuple[int, int]]],
              brain: np.ndarray) -> dict:
    """variants: (name, map, score, (dy,dx)). Shifted maps are shifted back; compared on a 4 px-cropped interior,
    restricted to the (dilated) brain when it has >= 100 px."""
    c = 4
    sl = (slice(c, -c), slice(c, -c))
    region = ndimage.binary_dilation(brain, iterations=5)[sl]
    if region.sum() < 100:
        region = np.ones_like(region)
    pk0 = np.unravel_index(int(np.argmax(base_map)), base_map.shape)
    rows, scores = [], [base_score]
    for name, m, sc, (dy, dx) in variants:
        back = np.roll(m, (-dy, -dx), axis=(0, 1))
        a, b = base_map[sl][region], back[sl][region]
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
        pk = np.unravel_index(int(np.argmax(back)), back.shape)
        rows.append({"name": name, "score": float(sc), "map_corr": corr,
                     "peak_shift_px": float(np.hypot(pk[0] - pk0[0], pk[1] - pk0[1]))})
        scores.append(float(sc))
    s = np.asarray(scores)
    return {"n_variants": len(rows), "variants": rows,
            "mean_map_corr": float(np.mean([r["map_corr"] for r in rows])),
            "min_map_corr": float(np.min([r["map_corr"] for r in rows])),
            "score_cv": float(s.std() / s.mean()) if s.mean() > 0 else float("inf"),
            "max_peak_shift_px": float(np.max([r["peak_shift_px"] for r in rows]))}


def location_sanity(amap: np.ndarray, brain: np.ndarray) -> dict:
    """Where does the top ``top_frac`` of the anomaly map sit: outside the brain / on the frame border?"""
    R = EVIDENCE_RULES
    k = max(1, int(np.ceil(R["top_frac"] * amap.size)))
    top = np.zeros(amap.shape, bool)
    top.flat[np.argpartition(amap.ravel(), -k)[-k:]] = True
    near = ndimage.binary_dilation(brain, iterations=3)
    b = R["border_px"]
    border = np.zeros_like(top)
    border[:b], border[-b:], border[:, :b], border[:, -b:] = True, True, True, True
    return {"top_pixels": int(k), "outside_brain_frac": float((top & ~near).sum() / k),
            "border_frac": float((top & border).sum() / k)}


def flagged_regions(amap: np.ndarray, pixel_threshold: float | None, max_regions: int = 5) -> list[dict]:
    """8-connected regions of amap >= pixel_threshold, largest first (area, bbox, max/mean score)."""
    if pixel_threshold is None:
        return []
    lab, n = ndimage.label(amap >= pixel_threshold, structure=np.ones((3, 3)))
    out = []
    for r, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab[sl] == r
        v = amap[sl][m]
        out.append({"area_px": int(m.sum()), "bbox_yxyx": [int(sl[0].start), int(sl[1].start), int(sl[0].stop), int(sl[1].stop)],
                    "max_score": float(v.max()), "mean_score": float(v.mean())})
    return sorted(out, key=lambda d: -d["area_px"])[:max_regions]


def assess_evidence(score: float, image_threshold: float | None, stab: dict, loc: dict) -> dict:
    """Explicit rules -> evidence strength in {not_flagged, strong, moderate, weak}. Heuristic signal only."""
    R = EVIDENCE_RULES
    if image_threshold is None:
        return {"flagged": None, "strength": "unavailable", "reasons": ["no frozen validation threshold for this model"],
                "margin_ratio": None}
    margin = (score - image_threshold) / image_threshold if image_threshold > 0 else float("inf")
    flagged = score >= image_threshold
    reasons = []
    if stab["mean_map_corr"] < R["min_mean_map_corr"]:
        reasons.append(f"anomaly map unstable under small perturbations (mean correlation {stab['mean_map_corr']:.3f} < {R['min_mean_map_corr']})")
    if stab["score_cv"] > R["max_score_cv"]:
        reasons.append(f"image score varies across perturbations (CV {stab['score_cv']:.3f} > {R['max_score_cv']})")
    if stab["max_peak_shift_px"] > R["max_peak_shift_px"]:
        reasons.append(f"peak location moves up to {stab['max_peak_shift_px']:.1f} px (> {R['max_peak_shift_px']})")
    if loc["outside_brain_frac"] > R["max_outside_brain_frac"]:
        reasons.append(f"{loc['outside_brain_frac']:.2f} of the strongest pixels lie outside the foreground region")
    if loc["border_frac"] > R["max_border_frac"]:
        reasons.append(f"{loc['border_frac']:.2f} of the strongest pixels lie on the frame border")
    if not flagged:
        return {"flagged": False, "strength": "not_flagged", "reasons": reasons, "margin_ratio": float(margin)}
    if reasons:
        strength = "weak"
    elif margin >= R["strong_margin_ratio"]:
        strength = "strong"
    else:
        strength = "moderate"
        reasons.append(f"score only {margin * 100:.1f}% above the validation-frozen threshold")
    return {"flagged": True, "strength": strength, "reasons": reasons, "margin_ratio": float(margin)}
