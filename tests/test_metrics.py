"""Metric correctness tests. Inputs are random numeric arrays (code-path fixtures, not images)."""
import numpy as np
import pytest
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

from brainsentinel.evaluation.metrics import (PixelStats, dice_iou_at, image_auroc_aupr,
                                              image_threshold_metrics, patient_bootstrap_ci)
from brainsentinel.evaluation.thresholding import (freeze, load_frozen, select_image_threshold,
                                                   select_pixel_threshold)


def _toy(n=24, h=32, seed=0, signal=1.5):
    """Maps = noise + signal inside square 'lesions' on half the images."""
    rng = np.random.default_rng(seed)
    maps = rng.normal(0, 1, (n, h, h)).astype(np.float32)
    masks = np.zeros((n, h, h), np.uint8)
    for i in range(0, n, 2):
        y, x = rng.integers(4, h - 12, 2)
        masks[i, y:y + 6, x:x + 7] = 1
        masks[i, y + 10:y + 12, x + 1:x + 3] = 1  # second, separate region on the same image
    maps = maps + signal * masks
    return maps, masks


def test_image_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = rng.normal(0, 1, 300) + y
    m = image_auroc_aupr(s, y)
    assert m["image_auroc"] == pytest.approx(roc_auc_score(y, s))
    assert m["image_aupr"] == pytest.approx(average_precision_score(y, s))


def test_image_metrics_need_both_classes():
    with pytest.raises(ValueError):
        image_auroc_aupr(np.arange(5.0), np.zeros(5))


def test_image_threshold_confusion_by_hand():
    s = np.array([0.1, 0.4, 0.6, 0.9])
    y = np.array([0, 1, 0, 1])
    m = image_threshold_metrics(s, y, 0.5)         # flags 0.6 (FP) and 0.9 (TP)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["specificity"] == 0.5
    assert m["fpr"] == 0.5 and m["f1"] == 0.5


def test_pixel_auroc_aupr_match_sklearn():
    maps, masks = _toy()
    st = PixelStats.build(maps, masks, n_bins=4096)
    assert st.pixel_auroc() == pytest.approx(roc_auc_score(masks.ravel(), maps.ravel()), abs=2e-3)
    assert st.pixel_aupr() == pytest.approx(average_precision_score(masks.ravel(), maps.ravel()), abs=5e-3)


def test_pixel_counts_are_exact():
    maps, masks = _toy()
    st = PixelStats.build(maps, masks)
    assert st.n_pos == int(masks.sum()) and st.n_neg == int((masks == 0).sum())


def test_perfect_and_inverted_maps():
    _, masks = _toy()
    perfect = masks.astype(np.float32)
    st = PixelStats.build(perfect, masks)
    assert st.pixel_auroc() == pytest.approx(1.0) and st.aupro() == pytest.approx(1.0)
    inv = PixelStats.build(1 - perfect, masks)
    assert inv.pixel_auroc() == pytest.approx(0.0, abs=1e-6)


def test_aupro_of_random_scorer_is_half_the_fpr_limit():
    rng = np.random.default_rng(3)
    masks = np.zeros((40, 48, 48), np.uint8)
    for i in range(0, 40, 2):
        masks[i, 10:20, 10:22] = 1
    maps = rng.random(masks.shape).astype(np.float32)
    assert PixelStats.build(maps, masks).aupro(0.3) == pytest.approx(0.15, abs=0.02)


def _brute_aupro(maps, masks, limit):
    """Exact per-threshold AUPRO using every distinct score as a threshold."""
    regions = []
    for i in range(len(maps)):
        lab, n = ndimage.label(masks[i], structure=np.ones((3, 3)))
        regions += [(i, lab == r) for r in range(1, n + 1)]
    neg = maps[masks == 0]
    thr = np.unique(maps)[::-1]
    fpr, pro = [0.0], [0.0]
    for t in thr:
        fpr.append(float((neg >= t).mean()))
        pro.append(float(np.mean([(maps[i][m] >= t).mean() for i, m in regions])))
    fpr, pro = np.array(fpr), np.array(pro)
    keep = fpr <= limit
    x, y = fpr[keep], pro[keep]
    j = int(keep.sum())
    if j < len(fpr):
        yl = y[-1] + (pro[j] - y[-1]) * (limit - x[-1]) / (fpr[j] - x[-1])
        x, y = np.append(x, limit), np.append(y, yl)
    return np.trapezoid(y, x) / limit


def test_aupro_matches_bruteforce():
    maps, masks = _toy(n=10, h=24)
    assert PixelStats.build(maps, masks, n_bins=8192).aupro(0.3) == pytest.approx(_brute_aupro(maps, masks, 0.3), abs=0.01)


def test_two_regions_on_one_image_are_counted_separately():
    """Region A (9 px) is found perfectly, region B (25 px) is invisible (ties with background).

    Per-region PRO(x) = (1 + x) / 2 -> AUPRO(0.3) = 0.5 + 0.5 * 0.15 = 0.575 (ties interpolate linearly,
    i.e. random tie-breaking). If the two regions were merged, PRO(x) = (9 + 25x) / 34 -> 0.375.
    """
    maps = np.zeros((1, 20, 20), np.float32)
    masks = np.zeros((1, 20, 20), np.uint8)
    masks[0, 1:4, 1:4] = 1
    masks[0, 10:15, 10:15] = 1
    maps[0, 1:4, 1:4] = 1.0
    st = PixelStats.build(maps, masks)
    assert sorted(st.region_size.tolist()) == [9, 25]
    assert st.aupro(0.3) == pytest.approx(0.575, abs=0.01)


def test_roi_restricts_pixels():
    maps, masks = _toy()
    roi = np.zeros(masks.shape, bool)
    roi[:, 8:24, 8:24] = True
    st = PixelStats.build(maps, masks, roi=roi)
    assert st.n_pos + st.n_neg == int(roi.sum())


def test_dice_iou_at_threshold_by_hand():
    maps = np.zeros((2, 4, 4), np.float32)
    masks = np.zeros((2, 4, 4), np.uint8)
    masks[0, :2, :2] = 1                          # 4 lesion px
    maps[0, :2, :1] = 1.0                         # detect 2 of them
    maps[0, 3, 3] = 1.0                           # +1 FP
    maps[1, 0, 0] = 1.0                           # FP on a normal image
    d = dice_iou_at(maps, masks, 0.5)
    assert d["dice_pooled"] == pytest.approx(2 * 2 / (2 * 2 + 2 + 2))     # tp2 fp2 fn2
    assert d["iou_pooled"] == pytest.approx(2 / 6)
    assert d["dice_per_image"] == pytest.approx(2 * 2 / (4 + 1 + 2))      # image 0 only: tp2 fp1 fn2
    assert d["normal_fp_px_rate"] == pytest.approx(1 / 16)


def test_select_pixel_threshold_maximises_validation_dice():
    maps, masks = _toy(signal=3.0)
    st = PixelStats.build(maps, masks)
    thr = select_pixel_threshold(st)
    best = dice_iou_at(maps, masks, thr)["dice_pooled"]
    for t in np.linspace(maps.min(), maps.max(), 25):
        assert dice_iou_at(maps, masks, t)["dice_pooled"] <= best + 0.01


def test_image_threshold_rules():
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([np.linspace(0, 1, 50), np.linspace(0.4, 1.4, 50)])
    for rule in ("max_f1", "youden", "sens_at_least"):
        t = select_image_threshold(s, y, rule)
        assert s.min() <= t <= s.max()
    assert image_threshold_metrics(s, y, select_image_threshold(s, y, "sens_at_least"))["sensitivity"] >= 0.95
    with pytest.raises(ValueError):
        select_image_threshold(s, y, "nope")


def test_freeze_refuses_test_split(tmp_path):
    with pytest.raises(ValueError):
        freeze(tmp_path / "t.json", split="test", image_threshold=0.1, image_rule="max_f1",
               pixel_threshold=0.2, pixel_rule="max_dice", n_images=5)
    freeze(tmp_path / "t.json", split="valid", image_threshold=0.1, image_rule="max_f1",
           pixel_threshold=0.2, pixel_rule="max_dice", n_images=83)
    assert load_frozen(tmp_path / "t.json")["selected_on"] == "valid"


def test_load_frozen_rejects_non_valid(tmp_path):
    (tmp_path / "t.json").write_text('{"selected_on": "test"}')
    with pytest.raises(ValueError):
        load_frozen(tmp_path / "t.json")


def test_patient_bootstrap_resamples_patients():
    rng = np.random.default_rng(0)
    pat = np.repeat([f"p{i}" for i in range(40)], 5)
    y = np.repeat(rng.integers(0, 2, 40), 5)
    s = rng.normal(0, 1, 200) + y
    lo, hi = patient_bootstrap_ci(s, y, pat, lambda a, b: roc_auc_score(b, a), n_boot=200)
    assert lo < roc_auc_score(y, s) < hi


def test_nonfinite_maps_rejected():
    maps, masks = _toy()
    maps[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        PixelStats.build(maps, masks)
