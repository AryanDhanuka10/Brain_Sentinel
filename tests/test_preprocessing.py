"""Day 3 preprocessing tests. Real BMAD train files only; every number is a Day 2 measured fact."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from brainsentinel.data.checks import list_images, split_dir
from brainsentinel.data.palette import corner_bg
from brainsentinel.data.preprocessing import (
    IMAGENET_MEAN, IMAGENET_STD, REPO, file_list_sha256, get_norm, load_index_image, to_network_input)


def _train(cfg, name):
    return split_dir(cfg, "train/good", "img") / name


def test_train_palette_is_256_monotone(train_pal):
    assert len(train_pal.colors) == 256 and train_pal.is_monotone()


def test_clean_image_index_unchanged(cfg, train_pal):
    from brainsentinel.data.palette import load_rgb
    p = _train(cfg, "00003_60.png")
    idx, brain, bg = load_index_image(p, train_pal)
    assert bg == 0 and idx.dtype == np.uint8 and idx.shape == (240, 240)
    assert np.array_equal(idx, train_pal.encode(load_rgb(p)))
    assert np.array_equal(brain, idx > 0)


def test_offset_background_removed(cfg, train_pal):
    idx, brain, bg = load_index_image(_train(cfg, "01548_77.png"), train_pal)  # Day 2: corners all 6
    assert bg == 6
    assert idx[0, 0] == idx[0, -1] == idx[-1, 0] == idx[-1, -1] == 0
    assert np.array_equal(idx > 0, brain)  # subtracting bg and thresholding agree
    assert 0.1 < brain.mean() < 0.5


def test_network_input_shape_dtype_range(cfg, train_pal):
    idx, _, _ = load_index_image(_train(cfg, "00003_60.png"), train_pal)
    x = to_network_input(idx, IMAGENET_MEAN, IMAGENET_STD)
    assert x.shape == (3, 240, 240) and x.dtype == np.float32 and np.isfinite(x).all()
    back = [x[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c] for c in range(3)]
    for b in back:  # gray replicated x3 and exactly idx/255: no equalisation, no rescaling
        assert np.allclose(b, idx / 255.0, atol=1e-5)
    assert min(float(b.min()) for b in back) >= -1e-5 and max(float(b.max()) for b in back) <= 1 + 1e-5


def test_preprocessing_is_deterministic(cfg, train_pal):
    p = _train(cfg, "01548_77.png")
    a = to_network_input(load_index_image(p, train_pal)[0], IMAGENET_MEAN, IMAGENET_STD)
    b = to_network_input(load_index_image(p, train_pal)[0], IMAGENET_MEAN, IMAGENET_STD)
    assert np.array_equal(a, b)


def test_zero_std_raises():
    with pytest.raises(ValueError):
        to_network_input(np.zeros((240, 240), np.uint8), IMAGENET_MEAN, (0.0, 0.0, 0.0))


def test_wrong_size_raises(cfg, train_pal):
    with pytest.raises(ValueError):
        load_index_image(_train(cfg, "00003_60.png"), train_pal, size=256)


def test_corrupt_png_raises(tmp_path, train_pal):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    with pytest.raises((OSError, ValueError)):
        load_index_image(bad, train_pal)


def test_unseen_colour_raises_in_pipeline(cfg, train_pal, tmp_path):
    from brainsentinel.data.palette import load_rgb
    rgb = load_rgb(_train(cfg, "00003_60.png"))
    used = set(train_pal.keys.tolist())
    cand = next((r, g, b) for r in range(256) for g in range(256) for b in (0, 1)
                if (r << 16 | g << 8 | b) not in used)
    rgb[100, 100] = cand
    p = tmp_path / "unseen.png"
    Image.fromarray(np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]), "RGBA").save(p)
    with pytest.raises(ValueError):  # must not be mapped to the nearest colour
        load_index_image(p, train_pal)


def test_bright_background_raises():
    idx = np.full((240, 240), 200, np.uint8)
    with pytest.raises(ValueError):
        corner_bg(idx)


def test_default_norm_is_fixed_imagenet_constants(pp):
    assert pp["normalize"]["mode"] == "imagenet"
    assert get_norm(pp) == (IMAGENET_MEAN, IMAGENET_STD)


def test_train_stats_file_if_present(cfg, pp):
    path = REPO / pp["normalize"]["train_stats_path"]
    if not path.exists():
        pytest.skip("run scripts/fit_norm_stats.py first")
    d = json.loads(path.read_text())
    assert d["fitted_on"] == "train/good" and d["n_images"] == 7500
    assert d["file_list_sha256"] == file_list_sha256(list_images(cfg, "train/good"))
    assert 0 < d["mean"] < 1 and 0 < d["std"] < 1
