#!/usr/bin/env bash
cd "$(git rev-parse --show-toplevel)" || exit 2
OUT=results/day1_final_check.txt; mkdir -p results results/day1; : > "$OUT"
P=0; F=0
ok()   { echo "[PASS] $1" | tee -a "$OUT"; P=$((P+1)); }
bad()  { echo "[FAIL] $1" | tee -a "$OUT"; F=$((F+1)); }
info() { echo "[INFO] $1" | tee -a "$OUT"; }
must() { local n=$1; shift; if "$@" >/dev/null 2>&1; then ok "$n"; else bad "$n"; fi; }
empty(){ local n=$1; shift; local o; o=$("$@" 2>/dev/null); if [ -z "$o" ]; then ok "$n"; else bad "$n -> $(echo "$o" | head -3 | tr '\n' ' ')"; fi; }

FORBID='^(data/raw|data/processed|brain/|[.]venv/|artifacts/checkpoints)|[.](nii|nii[.]gz|ckpt|pt|pth|npy|npz|faiss)$'
tracked_forbidden() { git ls-files | grep -E "$FORBID"; }
leaked()            { git ls-files --others --exclude-standard | grep -E '^(brain|data/raw|data/processed|artifacts/checkpoints)/'; }
bigfiles()          { git ls-files --others --exclude-standard -z | xargs -0 -r du -k | awk '$1>5000'; }
bigblobs()          { git rev-list --objects --all 2>/dev/null | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' 2>/dev/null | awk '$1=="blob" && $3>5000000'; }

echo "== Environment ==" | tee -a "$OUT"
must "python is 3.12" python -c 'import sys; assert sys.version_info[:2]==(3,12)'
must "torch 2.15.0 + torchvision 0.29.2, CUDA sees an RTX 4060" python -c 'import torch,torchvision as tv; assert torch.__version__.split("+")[0]=="2.15.0" and tv.__version__.split("+")[0]=="0.29.2" and torch.cuda.is_available() and "4060" in torch.cuda.get_device_name(0)'
must "pip check clean" pip check
must "running inside the project venv (brain)" bash -c '[ "${VIRTUAL_ENV##*/}" = "brain" ]'

echo "== Tests ==" | tee -a "$OUT"
if python -m pytest tests/test_config.py tests/test_day1_env.py tests/test_day1_data_contract.py -v > results/day1/pytest.txt 2>&1; then
  ok "pytest: $(tail -n1 results/day1/pytest.txt)"; else bad "pytest failed (see results/day1/pytest.txt)"; fi
must "data manifest matches files on disk" sha256sum --quiet -c data/MANIFEST.sha256

echo "== Evidence and documents ==" | tee -a "$OUT"
while IFS='|' read -r tag msg; do
  case $tag in PASS) ok "$msg";; FAIL) bad "$msg";; INFO) info "$msg";; esac
done < <(python - <<'PYEOF'
import hashlib, json, re
from pathlib import Path
import yaml

def out(tag, msg): print(f"{tag}|{msg}")
def t(name, cond, extra=""): out("PASS" if cond else "FAIL", name + (f" ({extra})" if extra else ""))
def load(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None
def read(p): return Path(p).read_text() if Path(p).exists() else ""

env = load("results/day1_env.json") or {}
steps = {k: v for k, v in env.items() if not k.startswith("_")}
t("check_env: 9/9 smoke tests PASS", len(steps) == 9 and all(v["status"] == "PASS" for v in steps.values()))

aud = load("results/day1_dataset_audit.json") or {}
exp = {"train/good": 7500, "valid/good": 39, "valid/Ungood": 44, "test/good": 640, "test/Ungood": 3075}
t("official split counts 7500/39/44/640/3075", {k: v.get("n_images") for k, v in aud.get("groups", {}).items()} == exp)
t("dataset audit has no warnings", aud.get("warnings") == [], str(aud.get("warnings")))
t("patient overlap train/valid/test = 0/0/10", aud.get("patient_overlap_counts") == {"train&valid": 0, "train&test": 0, "valid&test": 10})

ig = load("results/day1_integrity.json") or {}
t("integrity: 11,298 files hashed", ig.get("n_files_hashed") == 11298)
t("integrity: 0 corrupt or zero-byte", ig.get("corrupt_or_zero_byte", {}).get("count") == 0)
t("integrity: 0 duplicate groups inside a split", ig.get("identical_pixels_within_split", {}).get("groups") == 0)
t("integrity: cross-split duplicates are exactly the 10 documented", ig.get("identical_pixels_across_splits", {}).get("groups") == 10)
dp = load("results/day1_cross_split_duplicates.json") or {}
t("duplicates: 10 image-identical and 10 mask-identical", (dp.get("n_image_identical"), dp.get("n_mask_identical")) == (10, 10))
t("I/O benchmark recorded", load("results/day1_io_bench.json") is not None)

ex = load("results/day1_audit_extra.json")
t("audit_extra.json present (image_colour, slice_index)", bool(ex) and "image_colour" in ex and "slice_index" in ex)
if ex:
    c = ex["image_colour"]
    out("INFO", f"image colour: {c['n_unique_rgb_triples_overall']} unique RGB triples; top-256 cover {c['frac_pixels_in_top256_triples']:.4f}; median {c['median_unique_triples_per_image']}/image")
    out("INFO", "valid/test overlap by group: " + str({k: len(v) for k, v in ex["valid_test_overlap_by_group"].items()}))

cfg = read("configs/data.yaml"); yaml.safe_load(cfg)
t("data.yaml parses, ends with newline, one known_issues block", cfg.endswith("\n") and len(re.findall(r"^known_issues:", cfg, re.M)) == 1)

rd = read("data/README.md")
for f in ("data/README.md", "results/day1_notes.md"):
    bad = [i + 1 for i, l in enumerate(read(f).splitlines()) if "TBD" in l or "@@" in l]
    t(f"{f}: exists, no TBD/placeholder left", Path(f).exists() and not bad, f"lines {bad[:8]}" if bad else "")
t("data/README: license + BMAD citation", "CC BY-NC-SA" in rd and "2306.11876" in rd)
t("data/README: BraTS citation entries", bool(re.search(r"Menze|Bakas|Baid|2107\.02314", rd)))
t("data/README: download source (Drive URL) recorded", "drive.google.com" in rd)
mf = Path("data/MANIFEST.sha256")
t("manifest has 11,298 entries", mf.exists() and len(mf.read_text().splitlines()) == 11298)
t("data/README records the manifest SHA-256", mf.exists() and hashlib.sha256(mf.read_bytes()).hexdigest() in rd)

req = [l.split("#")[0].strip() for l in read("requirements.txt").splitlines()]
req = [l for l in req if l]
t("requirements.txt: every direct dependency pinned", bool(req) and all("==" in l for l in req), str([l for l in req if "==" not in l]))
lk = read("requirements-lock.txt")
t("requirements-lock.txt pins torch 2.15.0 + torchvision 0.29.2",
  bool(re.search(r"^torch==2\.15\.0", lk, re.M)) and bool(re.search(r"^torchvision==0\.29\.2", lk, re.M)))
t("top README states 'not a diagnosis'", "not a diagnosis" in read("README.md").lower())
PYEOF
)

echo "== Git hygiene ==" | tee -a "$OUT"
for p in data/raw/Brain data/processed artifacts/checkpoints/x/y.bin brain x.ckpt x.pt x.npy x.nii.gz; do
  must "git ignores $p" git check-ignore -q "$p"
done
empty "no forbidden paths tracked"                     tracked_forbidden
empty "no brain/ or data/raw paths about to be added"  leaked
empty "no untracked non-ignored file over 5 MB"        bigfiles
empty "no blob over 5 MB in Git history"               bigblobs
must  "raw BraTS2021 removed from data/raw"            test ! -e data/raw/BraTS2021
info  "core.fileMode=$(git config core.fileMode) (expected false on fuseblk)"

echo | tee -a "$OUT"; echo "PASS=$P FAIL=$F" | tee -a "$OUT"
if [ "$F" -eq 0 ]; then echo "DAY 1 GATE: PASS" | tee -a "$OUT"; else echo "DAY 1 GATE: FAIL - do not commit" | tee -a "$OUT"; fi
[ "$F" -eq 0 ]
