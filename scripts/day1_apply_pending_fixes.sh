#!/usr/bin/env bash
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2

# .gitignore: tidy trailing spaces, newline-safe append of the missing nii rules
sed -i 's/[ \t]*$//' .gitignore
[ -n "$(tail -c1 .gitignore)" ] && echo >> .gitignore
for l in '*.nii' '*.nii.gz'; do grep -qxF "$l" .gitignore || echo "$l" >> .gitignore; done

python - <<'PYEOF'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# requirements.txt: pin every unpinned direct dependency from what is installed
p = Path("requirements.txt")
out, missing = [], []
for line in p.read_text().splitlines():
    code, sep, comment = line.partition("#")
    name = code.strip()
    if name and "==" not in name and not name.startswith("-"):
        try:
            line = f"{name}=={version(name)}" + (f"  #{comment}" if sep else "")
        except PackageNotFoundError:
            missing.append(name)
    out.append(line)
if missing:
    print(f"requirements.txt NOT changed, not installed: {missing}  -> pip install them, rerun")
else:
    p.write_text("\n".join(out) + "\n")
    print("requirements.txt pinned")

# gate: manifest size = images + masks from the measured audit
g = Path("scripts/day1_final_check.sh"); s = g.read_text()
old = 't("manifest has 11,298 entries", mf.exists() and len(mf.read_text().splitlines()) == 11298)'
new = ('n_exp = sum(x.get("n_images", 0) + x.get("n_masks", 0) for x in aud.get("groups", {}).values())\n'
       't(f"manifest has one entry per image and mask ({n_exp})", mf.exists() and len(mf.read_text().splitlines()) == n_exp)')
if "n_exp" in s:
    print("gate already patched")
elif old in s:
    g.write_text(s.replace(old, new)); print("gate patched")
else:
    print("WARNING: gate manifest line not found")
PYEOF
echo "fixes applied"
