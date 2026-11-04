from pathlib import Path

import numpy as np
import pytest

from brainsentinel.data.checks import load_config
from brainsentinel.data.palette import brain_mask, corner_bg, train_palette

ROOT = Path(__file__).resolve().parents[1]


from brainsentinel.data.palette import Palette, load_rgb, pack_rgb

TRAIN = Path(__file__).resolve().parents[1] / "data" / "raw" / "Brain" / "train" / "good"
FILES = sorted(TRAIN.glob("*.png"))[:50]
pytestmark = pytest.mark.skipif(not FILES, reason="BMAD train/good not found")


@pytest.fixture(scope="module")
def imgs():
    return [load_rgb(p) for p in FILES]


@pytest.fixture(scope="module")
def pal(imgs):
    return Palette.from_keys(np.unique(np.concatenate([pack_rgb(i).ravel() for i in imgs])))


def test_bijection(pal):
    assert len(np.unique(pal.keys)) == len(pal.colors) <= 256
    assert (np.diff(pal.keys.astype(np.int64)) > 0).all()  # strictly sorted


def test_roundtrip_exact(pal, imgs):
    for rgb in imgs:
        idx = pal.encode(rgb)
        assert idx.dtype == np.uint8 and idx.shape == rgb.shape[:2]
        assert np.array_equal(pal.decode(idx), rgb)


def test_unseen_colour_raises(pal, imgs):
    used = set(pal.keys.tolist())
    cand = next((r, g, b) for r in range(256) for g in range(256) for b in (0, 1)
                if (r << 16 | g << 8 | b) not in used)
    bad = imgs[0].copy()
    bad[0, 0] = cand
    with pytest.raises(ValueError):
        pal.encode(bad)

@pytest.fixture(scope="module")
def full_pal():
    cfg = load_config(ROOT / "configs" / "data.yaml")
    cfg["root"] = str(ROOT / "data" / "raw" / "Brain")
    return train_palette(cfg)


def test_clean_image_mask_unchanged(full_pal):
    idx = full_pal.encode(load_rgb(TRAIN / "00003_60.png"))
    assert corner_bg(idx) == 0
    assert np.array_equal(brain_mask(idx), idx > 0)


def test_offset_image_background(full_pal):
    idx = full_pal.encode(load_rgb(TRAIN / "01548_77.png"))  # measured Day 2: corners all 6
    assert corner_bg(idx) == 6
    assert 0.1 < brain_mask(idx).mean() < 0.5  # loose; failing means the background is not uniform


def test_nonuniform_corners_raise(full_pal):
    idx = full_pal.encode(load_rgb(TRAIN / "00003_60.png")).copy()
    idx[0, 0] = 5
    with pytest.raises(ValueError):
        corner_bg(idx)