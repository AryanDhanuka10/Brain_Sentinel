"""One-off Day 3 patcher. Every target string must occur exactly once or NOTHING is written.
Run once from the repo root:  python scripts/patch_day3.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PALETTE = "src/brainsentinel/data/palette.py"
CONFTEST = "tests/conftest.py"

PATCHES = {
    PALETTE: [
        # 1) train-derived background tolerance constant
        ("from .checks import list_images",
         "from .checks import list_images\n\nMAX_BG = 10  # train-derived (Day 2): offset backgrounds are 1..10"),
        # 2) close the file handle in load_rgb
        ('    arr = np.asarray(Image.open(path).convert("RGBA"))',
         '    with Image.open(path) as im:\n        arr = np.asarray(im.convert("RGBA"))'),
        # 3) corner_bg refuses a background above the tolerance (bright corner => near-empty brain mask)
        ("    return c.pop()",
         "    v = c.pop()\n"
         "    if v > MAX_BG:\n"
         '        raise ValueError(f"corner background {v} exceeds train-derived tolerance {MAX_BG}")\n'
         "    return v"),
        # 4) train palette must be exactly 256 monotone colours (measured Day 2)
        ("    return Palette.from_keys(np.flatnonzero(seen))",
         "    pal = Palette.from_keys(np.flatnonzero(seen))\n"
         "    if len(pal.colors) != 256 or not pal.is_monotone():\n"
         '        raise ValueError(f"train palette must be 256 monotone colours, got {len(pal.colors)} '
         '(monotone={pal.is_monotone()})")\n'
         "    return pal"),
    ],
    CONFTEST: [
        ("    return c",
         "    return c\n\n\n"
         '@pytest.fixture(scope="session")\n'
         "def train_pal(cfg):\n"
         "    from brainsentinel.data.palette import train_palette\n"
         "    return train_palette(cfg)\n\n\n"
         '@pytest.fixture(scope="session")\n'
         "def pp():\n"
         "    from brainsentinel.data.preprocessing import load_pp_config\n"
         "    return load_pp_config()\n\n\n"
         '@pytest.fixture(scope="session")\n'
         "def index_rows():\n"
         "    import csv\n"
         '    p = REPO / "data" / "processed" / "index.csv"\n'
         "    if not p.exists():\n"
         '        pytest.skip("run scripts/build_index.py first")\n'
         '    with p.open(newline="") as f:\n'
         "        return list(csv.DictReader(f))\n"),
    ],
}


def main() -> None:
    staged = {}
    for rel, pairs in PATCHES.items():
        text = (REPO / rel).read_text()
        for old, new in pairs:
            n = text.count(old)
            if n != 1:
                sys.exit(f"ABORT (nothing written): {rel}: target occurs {n}x, expected 1: {old!r}")
            text = text.replace(old, new)
        compile(text, rel, "exec")  # syntax check before writing
        staged[rel] = text
    for rel, text in staged.items():
        (REPO / rel).write_text(text)
        print("patched", rel)


if __name__ == "__main__":
    main()
