"""Day 3 dataset tests. Real BMAD files only; numbers pinned to Day 1-2 measured facts."""
import csv

import numpy as np
import pytest
import torch

from brainsentinel.data.checks import split_dir
from brainsentinel.data.dataset import BMADBrain, assert_split_isolation, make_loader
from brainsentinel.data.preprocessing import get_norm, load_index_image, to_network_input

VALID = ["valid/good", "valid/Ungood"]
TEST = ["test/good", "test/Ungood"]


@pytest.fixture(scope="module")
def ds_train(cfg, pp, train_pal, index_rows):
    return BMADBrain(cfg, pp, train_pal, ["train/good"])


@pytest.fixture(scope="module")
def ds_valid(cfg, pp, train_pal, index_rows):
    return BMADBrain(cfg, pp, train_pal, VALID)


@pytest.fixture(scope="module")
def ds_test(cfg, pp, train_pal, index_rows):
    return BMADBrain(cfg, pp, train_pal, TEST)


def test_lengths(ds_train, ds_valid, ds_test):
    assert (len(ds_train), len(ds_valid), len(ds_test)) == (7500, 83, 3715)


def test_valid_item_shapes_and_dtypes(ds_valid):
    it = ds_valid[0]
    assert it["image"].shape == (3, 240, 240) and it["image"].dtype == torch.float32
    assert it["mask"].shape == (240, 240) and it["mask"].dtype == torch.uint8
    assert it["brain"].shape == (240, 240) and it["brain"].dtype == torch.bool
    assert {"patient", "slice", "slice_bin", "corner_bg", "key", "file"} <= set(it)
    assert isinstance(it["slice"], int) and torch.isfinite(it["image"]).all()


def test_train_item_has_no_mask(ds_train):
    assert "mask" not in ds_train[0] and ds_train[0]["key"] == "train/good"


def test_masks_binary_and_match_index_lesion_px(ds_valid, ds_test):
    # ASSUMPTION: index.csv lesion_px = number of foreground (RGB != 0) pixels. A failure here means that
    # assumption or the mask path is wrong; do not loosen the check.
    for ds, ids in ((ds_valid, range(len(ds_valid))), (ds_test, range(0, len(ds_test), 37))):
        for i in ids:
            m = ds[i]["mask"].numpy()
            assert set(np.unique(m).tolist()) <= {0, 1}
            assert int(m.sum()) == int(ds.items[i][1]["lesion_px"])


def test_sub100_lesions_survive(ds_test):
    ids = [i for i, (k, r) in enumerate(ds_test.items) if k == "test/Ungood" and int(r["lesion_px"]) < 100]
    assert len(ids) == 67  # Day 2 measured
    for i in ids:
        assert 1 <= int(ds_test[i]["mask"].sum()) < 100  # no mask resizing, nothing lost


def test_valid_lesions_inside_brain(ds_valid):
    for i in range(len(ds_valid)):  # Day 2 measured: 0 of 93,756 valid lesion px on background
        it = ds_valid[i]
        assert int((it["mask"].bool() & ~it["brain"]).sum()) == 0, it["file"]


def test_getitem_deterministic(ds_valid):
    a, b = ds_valid[3], ds_valid[3]
    assert torch.equal(a["image"], b["image"]) and torch.equal(a["mask"], b["mask"])


def test_loader_seed_controls_order(ds_train):
    first = lambda s: next(iter(make_loader(ds_train, 8, shuffle=True, seed=s)))["file"]
    assert first(0) == first(0) and first(0) != first(1)


def test_batch_roundtrip_against_files(cfg, pp, train_pal, ds_train):
    b = next(iter(make_loader(ds_train, 8, shuffle=True, seed=0)))
    assert b["image"].shape == (8, 3, 240, 240) and torch.isfinite(b["image"]).all()
    mean, std = get_norm(pp)
    for j, name in enumerate(b["file"]):
        idx, _, _ = load_index_image(split_dir(cfg, "train/good", "img") / name, train_pal)
        assert np.array_equal(b["image"][j].numpy(), to_network_input(idx, mean, std))


def test_isolation_passes_on_real_index(index_rows):
    assert_split_isolation(index_rows)


def test_isolation_detects_injected_overlap(index_rows):
    bad = dict(next(r for r in index_rows if r["split"] == "valid"))
    bad["patient"] = next(r["patient"] for r in index_rows if r["split"] == "train")
    with pytest.raises(ValueError):
        assert_split_isolation(index_rows + [bad])


def test_refuses_file_not_in_index(cfg, pp, train_pal, index_rows, tmp_path):
    drop = next(r for r in index_rows if r["split"] == "train")
    p = tmp_path / "index_missing_row.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0]))
        w.writeheader()
        w.writerows(r for r in index_rows if r is not drop)
    with pytest.raises(ValueError, match="mismatch"):
        BMADBrain(cfg, pp, train_pal, ["train/good"], index_csv=p)


def test_sensitivity_exclusion_removes_50(cfg, pp, train_pal, index_rows):
    ds = BMADBrain(cfg, pp, train_pal, TEST, exclude_patients=cfg["patient_overlap"]["valid_test_ids"])
    assert len(ds) == 3715 - 50


def test_offset_image_item(ds_train):
    i = next(i for i, (_, r) in enumerate(ds_train.items) if r["file"] == "01548_77.png")
    it = ds_train[i]
    assert it["corner_bg"] == 6
    assert abs(float(it["image"][0, 0, 0]) * 0.229 + 0.485) < 1e-5  # background is exactly 0 after removal
    assert 0.1 < float(it["brain"].float().mean()) < 0.5
