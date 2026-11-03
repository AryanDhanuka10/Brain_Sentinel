#!/usr/bin/env bash
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
rm -rf results/day1; mkdir -p results/day1          # drops stale evidence
python scripts/check_env.py                        2>&1 | tee results/day1/check_env.txt
python scripts/audit_dataset.py                    > results/day1/audit_dataset_stdout.txt 2>&1
python scripts/audit_extra.py                      > results/day1/audit_extra_stdout.txt 2>&1
python scripts/audit_integrity.py                  > results/day1/audit_integrity_stdout.txt 2>&1
python scripts/diagnose_cross_split_duplicates.py  > results/day1/cross_split_duplicates_stdout.txt 2>&1
python scripts/bench_io.py                         > results/day1/bench_io_stdout.txt 2>&1
bash   scripts/day1_git_check.sh                   > /dev/null 2>&1
pip freeze > requirements-lock.txt
find data/raw/Brain -type f -name "*.png" -print0 | sort -z | xargs -0 sha256sum > data/MANIFEST.sha256
wc -l data/MANIFEST.sha256
echo "done. evidence in results/ and results/day1/"
