#!/usr/bin/env bash
# Read-only Git hygiene check -> results/day1_git_check.txt
set +e
OUT=results/day1_git_check.txt; mkdir -p results; : > "$OUT"
say() { echo "$@" | tee -a "$OUT"; }
say "== .gitignore =="; cat .gitignore | tee -a "$OUT"
say "== check-ignore (each path must show a rule) =="
for p in data/raw/Brain data/raw/BraTS2021 data/processed artifacts/checkpoints brain .venv x.ckpt x.pt x.npy x.nii.gz; do
  r=$(git check-ignore -v "$p" 2>&1); [ -n "$r" ] && say "IGNORED      $r" || say "NOT IGNORED  $p"
done
say "== tracked files that must NOT be tracked (expect empty) =="
git ls-files | grep -E '^(data/raw|data/processed|brain/|\.venv/|artifacts/checkpoints)|\.(nii|nii\.gz|ckpt|pt|pth|npy|npz|faiss)$' | head -20 | tee -a "$OUT"
say "== blobs >5MB anywhere in history (expect empty) =="
git rev-list --objects --all 2>/dev/null | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' 2>/dev/null | awk '$1=="blob" && $3>5000000' | head -20 | tee -a "$OUT"
say "== untracked, not-ignored files >5MB (expect empty) =="
git ls-files --others --exclude-standard -z | xargs -0 -r du -k 2>/dev/null | awk '$1>5000' | head | tee -a "$OUT"
say "== status entries =="; git status --short | wc -l | tee -a "$OUT"
say "== data/raw contents (BraTS2021 should be gone) =="; du -sh data/raw/* 2>/dev/null | tee -a "$OUT"
say "== core.fileMode =="; git config core.fileMode | tee -a "$OUT"
