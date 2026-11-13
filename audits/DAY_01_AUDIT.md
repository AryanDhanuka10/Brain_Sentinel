# Day 1 - Research, setup, dataset download (retrospective)

## 1. Original objectives
- Working environment, GPU visible, BMAD Brain downloaded from the official link, data/README.md, no raw data in Git.

## 2. Implemented functionality
- Environment (torch 2.15.0 / torchvision 0.29.2, RTX 4060), dataset, manifest, README with source/licence/citations.

## 3. Files created or modified
- data/README.md (modified), tests/test_day1_env.py (modified), requirements.txt, pyproject.toml (was empty)

## 4. Technical review
- Evidence reviewed, not re-executed here (no GPU/dataset in sandbox): results/day1/ (check_env 9/9, 41 tests at the time). The torch pins reproduce in the sandbox. Defects found: data/README.md ended with a pasted heredoc terminator 'EOF' and cited a non-existent path; pyproject.toml empty; Day-1 hardware tests hard-fail on any machine without the exact GPU (breaks CPU fallback/CI); requirements.txt lacked all later dependencies.

## 5. Tests executed
- User-run suite and logs under results/ (not re-executed here: needs GPU/dataset); in the sandbox only the tests named under 'Actual test results' were run.

## 6. Actual test results
- Sandbox: tests/test_day1_env.py -> pinned-version test passes, 3 hardware tests skip without CUDA (assert fully when CUDA exists). test_repo_hygiene.py: 4 passed.

## 7. Bugs discovered
- README 'EOF'; wrong result path; empty pyproject; hardware tests not CPU-safe; missing deps.

## 8. Fixes applied
- All fixed with regression tests (test_repo_hygiene.py); requirements extended with versions verified by clean install.

## 9. Remaining issues
- Re-run `pytest tests/test_day1_env.py` on the RTX 4060 laptop and paste output.

## 10. Acceptance decision
- CONDITIONAL PASS

## 11. Evidence and artifact references
- results/day1/*
- data/MANIFEST.sha256

