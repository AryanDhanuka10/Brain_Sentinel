"""Regression tests for documentation/packaging defects found in the Day 0-3 audit."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_data_readme_has_no_heredoc_terminator():
    text = (REPO / "data" / "README.md").read_text()
    assert not text.rstrip().endswith("EOF")
    assert text.endswith("\n")


def test_data_readme_result_paths_exist():
    text = (REPO / "data" / "README.md").read_text()
    for rel in re.findall(r"results/[\w./-]+\.(?:md|json|txt)", text):
        assert (REPO / rel).exists(), f"data/README.md references missing file {rel}"


def test_pyproject_is_not_empty_and_declares_package():
    text = (REPO / "pyproject.toml").read_text()
    assert 'name = "brainsentinel"' in text and "[tool.pytest.ini_options]" in text


def test_inspect_script_has_no_misleading_corner_key():
    text = (REPO / "scripts" / "inspect_dataset.py").read_text()
    assert "images_with_nonblack_corner" not in text
