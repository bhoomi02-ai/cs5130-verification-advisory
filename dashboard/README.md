# Results dashboard (static web page)

A single-page, read-only view of the study results. It never calls a model, never reads the answer key and never spends money: the page only loads small JSON files that `build_data.py` derives from `results/`.

> Work in progress. Step 2 of 4 is done (the data files). The page itself (views 1–9) comes next.

## Regenerate the data

```bash
python dashboard/build_data.py                                   # results/ -> dashboard/data/
python dashboard/build_data.py --results path/to/results         # e.g. results extracted from study-results
```

`dashboard/data/` is **gitignored on `main`** (it is derived from the study results, which live on the `study-results` branch). Re-run the script after every batch; it replaces the folder.

To get the results on a fresh clone first:

```bash
git fetch origin study-results
git archive origin/study-results results | tar -x
python dashboard/build_data.py
```

## What the script writes

| File | Contents | Built from |
|---|---|---|
| `meta.json` | Study settings (model, temperature, top-p, tasks × repeats), each setting's progress, which settings are complete, tasks whose scoring is under review | `results.jsonl`, `data/study_tasks.txt` |
| `summary.json` | Headline sentence; per setting: task success with 95% CI and counts, success without human review, human-review routes, first-draft success, kept / fixed / broken / still wrong, catch rate and false-alarm rate with counts, MCC, verifier calls, cost and tokens; paired comparisons vs `none` | `metrics.summarize`, `metrics.compare_conditions` (the same code as `scripts/analyze.py`) |
| `heatmap.json` | Right final answers out of saved runs per task and setting, grouped under S01–S10 with the variants as sub-rows | `results.jsonl` |
| `spend.json` | Total spend vs the $20 cap; by setting, model and day | `runs.jsonl` (every call once, as `scripts/spend.py`) |
| `seeded.json` | Seeded-set catch rate, false alarms and "caught with the right reason" per setting, and every seeded review's verdict and decision | `results/seeded/`, `right_reason.py` |
| `sanity_s04.json` | The pre-study verifier checks on a known-wrong and a known-right S04 answer | `runs.jsonl` (`*-check` calls) |
| `runs.json` | One row per run, in the order of each run's first logged call (for the replay and the run table) | `results.jsonl`, `runs.jsonl` |
| `runs/<run_id>.json` | One run's draft, reviews (issues, reason, calculator use), revision, final answer, and the checks it failed | `results/answers/` |

## Rules the script enforces (and the tests check)

- **Pending, never a number.** A setting is complete only when every study run is saved and none failed. Unfinished settings are exported as `{"status": "pending", "done": 54, "total": 96, "label": "pending (54/96)"}` with no rates; the same applies to each setting's seeded reviews (31 answers).
- **No answer key.** The checker's `expected` values and the seeded labels' `correct_value` / `wrong_value` are dropped from every file, at any depth. Failed checks show only what the answer reported (`observed`) and the reason code (e.g. `value_mismatch`). The seeded labels contribute only the right-reason decision.
- **No account identifiers.** API keys, Groq organization IDs, e-mail addresses and home-folder paths are redacted from every string (e.g. Groq's rate-limit errors carry the organization ID).
- **Sample sizes everywhere.** Every rate comes with its counts (for example `{"value": 1.0, "caught": 11, "n": 11}`) and task success with its 95% bootstrap CI over the 32 tasks.
- **Spend** is the total of `runs.jsonl`. A run's own `cost_usd` includes its share of the shared first draft, so per-run costs are shown as recorded and labelled that way.
- **Human review** has three routes: an abstain, an unreadable verdict, or a reject whose revision the second check did not accept. Task success is reported both with and without those runs.
- **T06-v003** is graded as the checker has it, with a "scoring under review" note (`SCORING_UNDER_REVIEW` in `src/advisory/dashboard_export.py`).

Tests: `tests/test_dashboard_export.py` builds from a tiny fixture that contains answer-key values and account identifiers on purpose, and fails if any of them reach the output.

## Mockup

`mockup/dashboard_figma.svg` (and a PNG render) is the concept design: dark theme, violet `#8B5CF6` to cyan `#22D3EE` on `#07060F`, pink `#FF5C8A` for wrong or broken. Its numbers are placeholders from the 10 × 5 pilot baseline; the page will show the main study (32 tasks × 3 runs).
