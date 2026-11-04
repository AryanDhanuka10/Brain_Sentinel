#!/usr/bin/env bash
cd "$(git rev-parse --show-toplevel)" || exit 2
echo "### DAY 1 FULL REPORT  $(date -Is)"
echo "### branch=$(git rev-parse --abbrev-ref HEAD 2>&1)  commit=$(git rev-parse --short HEAD 2>&1)"
echo "### python=$(which python)  $(python --version 2>&1)"

echo; echo "### [1] regenerate evidence (log: results/day1_run_all.log)"
bash scripts/day1_run_all.sh > results/day1_run_all.log 2>&1; echo "run_all exit=$?"
python scripts/audit_palette.py > results/day1/palette_stdout.txt 2>&1; echo "palette exit=$?"
python scripts/day1_make_notes.py > /dev/null 2>&1; echo "notes exit=$?"

echo; echo "### [2] GATE"
bash scripts/day1_final_check.sh; echo "gate exit=$?"

echo; echo "### [3] pytest tail"; tail -n 3 results/day1/pytest.txt

echo; echo "### [4] manifest breakdown (expect 15096 total)"
wc -l data/MANIFEST.sha256
python - <<'PYEOF'
from collections import Counter
c = Counter()
for line in open("data/MANIFEST.sha256"):
    p = line.split(None, 1)[1].strip().split("/Brain/", 1)[1].split("/")
    c["/".join(p[:2]) + ("/" + p[2] if len(p) == 4 else "")] += 1
for k, v in sorted(c.items()):
    print(f"{v:6d}  {k}")
PYEOF

echo; echo "### [5] slice-index histograms (per 10)"
python -c "import json;r=json.load(open('results/day1_audit_extra.json'));[print(k,v['hist_by_10']) for k,v in r['slice_index'].items()]"

echo; echo "### [6] palette check"; cat results/day1_palette_check.json

echo; echo "### [7] key lines of day1_notes.md"
grep -E "Smoke tests|Audit warnings|Colour structure|Mean slice index|lesion px|Files hashed|Identical-pixel" results/day1_notes.md | cut -c1-230

echo; echo "### [8] repo state"
echo "-- data/raw:"; ls data/raw; du -sh data/raw/* 2>/dev/null
echo "-- manifest sha256: $(sha256sum data/MANIFEST.sha256 | cut -d' ' -f1)"
echo "-- README records it: $(grep -c "$(sha256sum data/MANIFEST.sha256 | cut -d' ' -f1)" data/README.md)"
echo "-- requirements.txt (non-comment):"; grep -v '^\s*#' requirements.txt | grep -v '^\s*$'
echo "-- git status:"; git status --short
echo "-- ignore rules:"; git check-ignore -v x.nii.gz data/raw/Brain brain 2>&1 | cut -c1-90
echo "-- last commits / tags:"; git log --oneline -3 2>&1; git tag
echo; echo "### END OF REPORT"
