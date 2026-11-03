import hashlib
import random

import numpy as np
import pytest
from PIL import Image

from brainsentinel.data.checks import list_images, mask_path, mask_to_binary, parse_name, split_dir

KEYS = ["train/good", "valid/good", "valid/Ungood", "test/good", "test/Ungood"]
MASKED = KEYS[1:]
DARK = (56, 55, 77)


def _patients(cfg, split):
    return {parse_name(p.name)[0] for k in KEYS if k.startswith(split) for p in list_images(cfg, k)}


def _mask(cfg, key, p):
    return np.asarray(Image.open(mask_path(cfg, key, p)))


@pytest.mark.parametrize("key", KEYS)
def test_official_split_counts(cfg, key):
    assert len(list_images(cfg, key)) == cfg["splits"][key]["expected_n"]


@pytest.mark.parametrize("key", KEYS)
def test_filenames_parse(cfg, key):
    for p in list_images(cfg, key):
        parse_name(p.name)


@pytest.mark.parametrize("key", MASKED)
def test_image_mask_pairing(cfg, key):
    imgs = {p.name for p in list_images(cfg, key)}
    masks = {p.name for p in split_dir(cfg, key, "mask").glob("*.png")}
    assert imgs == masks


@pytest.mark.parametrize("key", KEYS)
def test_image_format(cfg, key):
    files = list_images(cfg, key)
    for p in random.Random(0).sample(files, min(30, len(files))):
        with Image.open(p) as im:
            assert im.size == (cfg["image_size"],) * 2 and im.mode == "RGBA"


@pytest.mark.parametrize("key", MASKED)
def test_masks_use_only_known_colours(cfg, key):
    allowed = {tuple(c) for c in cfg["mask"]["allowed_rgb"]}
    files = list_images(cfg, key)
    for p in random.Random(0).sample(files, min(300, len(files))):
        a = _mask(cfg, key, p)
        assert a.shape == (240, 240, 4) and (a[..., 3] == 255).all()
        assert {tuple(c) for c in np.unique(a[..., :3].reshape(-1, 3), axis=0).tolist()} <= allowed


@pytest.mark.parametrize("key", ["valid/good", "test/good"])
def test_good_masks_are_empty(cfg, key):
    for p in list_images(cfg, key):
        assert mask_to_binary(_mask(cfg, key, p)).sum() == 0, p.name


@pytest.mark.parametrize("key", ["valid/Ungood", "test/Ungood"])
def test_ungood_masks_are_nonempty(cfg, key):
    for p in list_images(cfg, key):
        assert mask_to_binary(_mask(cfg, key, p)).sum() > 0, p.name


def test_binarization_keeps_dark_lesion_labels(cfg):
    for p in list_images(cfg, "valid/Ungood"):
        a = _mask(cfg, "valid/Ungood", p)
        dark = (a[..., :3] == DARK).all(-1)
        if dark.any():
            b = mask_to_binary(a)
            assert b[dark].all() and set(np.unique(b).tolist()) == {0, 1}
            assert (a[..., :3].max(-1) > 127).sum() < b.sum()  # a '>127' rule would drop labels
            return
    pytest.skip("no valid/Ungood mask contains the dark label colour")


def test_train_patients_disjoint_from_valid_and_test(cfg):
    tr = _patients(cfg, "train")
    assert not (tr & _patients(cfg, "valid")) and not (tr & _patients(cfg, "test"))


def test_valid_test_overlap_is_the_documented_set(cfg):
    got = _patients(cfg, "valid") & _patients(cfg, "test")
    assert got == set(cfg["patient_overlap"]["valid_test_ids"])


def test_good_and_ungood_filenames_disjoint_within_split(cfg):
    for s in ("valid", "test"):
        g = {p.name for p in list_images(cfg, f"{s}/good")}
        u = {p.name for p in list_images(cfg, f"{s}/Ungood")}
        assert not (g & u)


def _names(cfg, split):
    return {p.name for k in KEYS if k.startswith(split) for p in list_images(cfg, k)}


def test_cross_split_filename_collisions_are_exactly_the_documented_duplicates(cfg):
    train, valid, test = (_names(cfg, s) for s in ("train", "valid", "test"))
    assert not (train & valid) and not (train & test)
    assert (valid & test) == set(cfg["known_issues"]["valid_test_duplicate_files"])


def test_documented_duplicates_are_byte_identical_ungood_images(cfg):
    for name in cfg["known_issues"]["valid_test_duplicate_files"]:
        a = split_dir(cfg, "valid/Ungood", "img") / name
        b = split_dir(cfg, "test/Ungood", "img") / name
        assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()


def test_sensitivity_exclusion_set_size(cfg):
    ids = set(cfg["patient_overlap"]["valid_test_ids"])
    n = sum(parse_name(p.name)[0] in ids for k in ("test/good", "test/Ungood") for p in list_images(cfg, k))
    assert n == cfg["known_issues"]["sensitivity_exclusion_expected_test_images"]
