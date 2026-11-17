# Study results (backup)

This branch holds **only run data**, no code. It's a backup of `results/` from the machine that makes the paid study runs, since `results/` is gitignored on `main`. **Never merge it into `main`.** It's updated after each batch.

## What's here (as of Nov 17)
| Path | What it holds |
|---|---|
| `results/results.jsonl` | One row per graded run: the **`none` baseline**, gemini-3.6-flash, S01–S10 × 5 (50 runs, 84% correct) |
| `results/drafts/gemini-3.6-flash/` | The saved first drafts those runs used (reused by every condition) |
| `results/report/` | `scripts/analyze.py` output for the baseline: summary, per-scenario and infeasible tables, chart |
| `results/runs.jsonl` | Every logged model call so far (pilot, baseline, live verifier checks): tokens, cost, latency, settings, raw output |
| `results/pilot/` | Model pilot batches (see `docs/pilot.md` on `main`) |

Checked before pushing: no API keys, account or project IDs, emails or local paths.

## Using it
Copy the files into your checkout's `results/` (they're gitignored there, so nothing gets committed by accident):
```bash
git fetch origin study-results
git archive origin/study-results results | tar -x
python scripts/analyze.py          # rebuilds results/report/ from results/results.jsonl
```
