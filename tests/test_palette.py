from pathlib import Path

import numpy as np
import pytest

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
