from pathlib import Path

import yaml

CFG = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"


def _load() -> dict:
    return yaml.safe_load(CFG.read_text())


def test_data_yaml_ends_with_newline():
    assert CFG.read_bytes().endswith(b"\n")


def test_data_yaml_parses_and_has_required_keys():
    assert {"root", "image_size", "splits", "mask", "patient_overlap", "known_issues"} <= set(_load())


def test_known_duplicates_match_overlap_patients():
    c = _load()
    dups = c["known_issues"]["valid_test_duplicate_files"]
    assert len(dups) == len(set(dups)) == 10
    assert set(dups) == {f"{i}_60.png" for i in c["patient_overlap"]["valid_test_ids"]}
