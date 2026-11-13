"""Regression tests for script defects found in the audit (import order, output locations)."""
import ast
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))


def _first_lines(tree):
    imp = [n.lineno for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
           and ((isinstance(n, ast.ImportFrom) and (n.module or "").startswith("brainsentinel"))
                or (isinstance(n, ast.Import) and any(a.name.startswith("brainsentinel") for a in n.names)))
           and n.col_offset == 0]
    src = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "insert" and "sys.path" in ast.unparse(n.func.value)]
    return (min(imp) if imp else None), (min(src) if src else None)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_adds_src_to_path_before_importing_package(script):
    tree = ast.parse(script.read_text())
    imp, ins = _first_lines(tree)
    if imp is None:
        return
    assert ins is not None and ins < imp, f"{script.name}: imports brainsentinel (line {imp}) before sys.path.insert"


def test_day2_scripts_write_into_results_day2():
    for name in ("build_index", "check_background", "check_palette_roundtrip", "diagnose_day2", "inspect_dataset"):
        text = (Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py").read_text()
        assert '"results" / "day2"' in text, name


def test_train_autoencoder_cli_does_not_inject_seed_hparam():
    import inspect
    from brainsentinel.models.autoencoder import ConvAutoencoder
    assert "seed" not in inspect.signature(ConvAutoencoder.__init__).parameters
    src = (SCRIPTS[0].parent / "train.py").read_text()
    assert 'if "seed" in mc["hparams"]' in src


def test_train_logs_interrupt_as_not_ok():
    src = (SCRIPTS[0].parent / "train.py").read_text()
    assert "except BaseException" in src and "INTERRUPTED" in src


def test_pca_script_only_reads_explained_variance_for_pca_variants():
    src = (SCRIPTS[0].parent / "pca_experiments.py").read_text()
    assert 'tag.startswith("pca")' in src and '"pca" in tag' not in src
