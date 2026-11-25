# Follow-up experiment (not part of the main study)

14 study-quality variants that were left out of `data/study_tasks.txt` for budget (`extension_tasks.txt`),
3 runs each, under `none`, `self`, `dedicated` and `weakened`. Same advisor and settings as the main
study (gemini-3.6-flash, temperature 0.7, top-p 0.9, advisor-v1), run on 2026-11-25 from main at the
commit with the S06 scoring fix (#63), so grading matches `results_regraded.jsonl`. Run in a separate
checkout so the main study's `results/` was never touched. Cost: $3.20.

Tables: `results/report/` (`python scripts/analyze.py --results extension/results/results.jsonl --out ...`).
