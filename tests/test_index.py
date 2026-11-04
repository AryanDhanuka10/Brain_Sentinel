import csv
from collections import Counter
from pathlib import Path

import pytest

from brainsentinel.data.checks import load_config
from brainsentinel.data.index import columns, size_bin, slice_bin

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "processed" / "index.csv"
needs_index = pytest.mark.skipif(not INDEX.exists(), reason="run scripts/build_index.py first")


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs" / "data.yaml")


@pytest.fixture(scope="module")
def rows():
    with INDEX.open() as f:
        return list(csv.DictReader(f))


def key(r):
    return f"{r['split']}/{r['class']}"


def test_slice_bin():
    assert slice_bin(60) == "60s" and slice_bin(69) == "60s" and slice_bin(99) == "90s"


def test_size_bin_edges():
    cases = {None: "na", 0: "0", 1: "1-99", 99: "1-99", 100: "100-999", 999: "100-999",
             1000: "1000-2999", 2999: "1000-2999", 3000: "3000+"}
    for px, want in cases.items():
        assert size_bin(px) == want


@needs_index
def test_columns(cfg, rows):
    assert set(rows[0]) == set(columns(cfg))


@needs_index
def test_counts_match_config(cfg, rows):
    want = {k: v["expected_n"] for k, v in cfg["splits"].items()}
    assert dict(Counter(key(r) for r in rows)) == want
    assert len(rows) == sum(want.values())


@needs_index
def test_masks_pairing_and_blank(rows):
    for r in rows:
        if r["split"] == "train":
            assert r["lesion_px"] == ""
        elif r["class"] == "good":
            assert int(r["lesion_px"]) == 0
        else:
            assert int(r["lesion_px"]) > 0


@needs_index
def test_dims_and_colours(rows):
    assert all(r["dims_ok"] == "1" for r in rows)
    assert all(r["mask_missing"] == "0" for r in rows)
    assert all(r["unknown_mask_px"] in ("", "0") for r in rows)


@needs_index
def test_train_patients_disjoint(rows):
    tr = {r["patient"] for r in rows if r["split"] == "train"}
    other = {r["patient"] for r in rows if r["split"] != "train"}
    assert not (tr & other)


@needs_index
def test_overlap_matches_config(cfg, rows):
    ids = {r["patient"] for r in rows if r["overlap_patient"] == "1"}
    assert ids == set(cfg["patient_overlap"]["valid_test_ids"])
    n_test = sum(1 for r in rows if r["split"] == "test" and r["overlap_patient"] == "1")
    assert n_test == cfg["known_issues"]["sensitivity_exclusion_expected_test_images"]


@needs_index
def test_cross_split_duplicates(cfg, rows):
    want = set(cfg["known_issues"]["valid_test_duplicate_files"])
    flagged = [r for r in rows if r["dup_cross_split"] == "1"]
    assert {r["file"] for r in flagged if r["split"] == "valid"} == want
    assert len(flagged) == 2 * len(want)
    assert not any(r["split"] == "train" for r in flagged)

@needs_index
def test_offset_background_pinned(rows):
    off = [r for r in rows if int(r["corner_bg"]) > 0]
    assert len(off) == 33  # measured Day 2
    assert {r["patient"] for r in off} == {"00584", "01548", "01551", "00446"}
    assert max(int(r["corner_bg"]) for r in off) == 10
    assert not any(r["split"] == "valid" for r in off)
