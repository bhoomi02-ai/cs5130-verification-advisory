# Run pipeline

How the study's runs were executed and recorded: one script per condition, one shared first draft per task and repeat, one row per run, and a runner that can be stopped and restarted without losing or duplicating anything. No agent ever sees the answer key, and every model call goes through `advisory.llm.call_model`.

> Section 3 of the report (System architecture and pipeline) and its Figure 1 describe the same flow.

## Run it

```bash
# One condition of the main study (PAID calls: run only on the paid-run machine)
python scripts/run_condition.py --condition none --model gemini-3.6-flash \
    --tasks-file data/study_tasks.txt --repeats 3

# Run the runs saved as failed again, replacing their rows
python scripts/run_condition.py --condition cross_family --model gemini-3.6-flash \
    --tasks-file data/study_tasks.txt --repeats 3 --retry-failed

# Seeded set: one condition's verifier reviews every answer in data/injected/
python scripts/run_seeded.py --condition dedicated --model gemini-3.6-flash --dry-run

# Spend so far, and the completeness check of the saved data
python scripts/spend.py
python scripts/check_study_data.py
```

`cross_family` needs `GROQ_API_KEY` in `.env`; its verifier is `openai/gpt-oss-20b`, while `--model` stays the advisor's model.

| Option (`run_condition.py`) | Meaning |
|---|---|
| `--condition` | `none`, `self`, `peer`, `dedicated`, `weakened` or `cross_family` |
| `--model` | The advisor's exact model name, never an alias |
| `--tasks` / `--tasks-file` | Task IDs, or a file with one per line (`data/study_tasks.txt` for the study) |
| `--repeats` | Runs per task (default 1; the study used 3) |
| `--temperature`, `--top-p`, `--thinking-level` | Generation settings (defaults 0.7 and 0.9; the study used the model's default thinking level) |
| `--retry-failed` | Run the runs saved as failed again, replacing their rows |

## How one run works

A run is one task, one repeat and one condition. `pipeline.run_one` takes four steps:

| Step | What happens | Code |
|---|---|---|
| 1. Shared first draft | The advisor's draft for (model, task, repeat) is written once, saved under `results/drafts/<model>/`, and reused by every condition | `drafts.py` |
| 2. Verification | The condition's verifier accepts, rejects or abstains; after a reject the advisor revises once and the same verifier checks the revision. `none` skips this step | `conditions.py`, `dedicated.py` |
| 3. Grading | The hidden checker grades the first draft and the final answer | `checker.py` |
| 4. Recording | One row is appended to `results/results.jsonl`; the draft, revision, verifier replies and both checker results go to `results/answers/<run_id>.json` | `pipeline.py` |

Run IDs look like `dedicated-gemini-3.6-flash-S04-r2`; the shared draft behind it is `draft-gemini-3.6-flash-S04-r2`.

Every run ends with one status:

| Status | Meaning |
|---|---|
| `unchecked` | `none`: the draft is the final answer |
| `accepted` | The verifier accepted the draft |
| `revised` | Rejected, revised once, and the revision was accepted |
| `human_review` | An abstain, an unreadable verdict, or a revision the second check did not accept |
| `failed` | An API call still failed after retries; the run is left out of every rate |

**The calculator** (`dedicated` and `cross_family` only) evaluates arithmetic plus `abs`, `round`, `min`, `max`, `sum`, `sqrt`, `log`, `log10`, `exp`, `ceil`, `floor` and the numpy-financial functions `fv`, `pv` and `pmt`. Expressions are parsed against a whitelist, so nothing the model sends runs as code. A review gets at most 3 rounds of up to 15 expressions (`calculator.py`, `prompts.MAX_CALCULATOR_ROUNDS`, `prompts.MAX_EXPRESSIONS_PER_ROUND`). `weakened` uses the same role and review instructions without the calculator.

## What each row records

`results/results.jsonl`, one JSON object per run:

| Fields | Meaning |
|---|---|
| `run_id`, `draft_run_id`, `task_id`, `condition`, `run_number` | Which run |
| `model`, `temperature`, `top_p`, `thinking_level` | Advisor settings |
| `verifier_model`, `verifier_thinking_level`, `verifier_prompt_version`, `revision_prompt_version`, `prompt_version` | Verifier settings and prompt versions |
| `initial_correct`, `final_correct`, `initial_status_correct`, `final_status_correct`, `initial_format_failure`, `final_format_failure` | Grades of the first draft and the final answer |
| `verdict`, `second_verdict`, `verifier_calls`, `status`, `error` | What the verifier did (`verdict` is always the first verdict) |
| `tokens_in`, `tokens_out`, `cost_usd`, `latency_s`, `n_calls`, `n_failed_calls` | Usage of the run's logged calls, including the shared draft |

## Failures, reruns and the settings guard

- **A failed call is saved, not lost.** An API error after `call_model`'s retries saves the run with status `failed` and its error, and the script moves on.
- **Three failures in a row stop the script** (`MAX_FAILED_IN_A_ROW = 3`), since that points to an outage rather than the condition.
- **A daily quota or a code error stops the script at once and saves nothing** for that run, so a setup problem is never recorded as a model failure. Run the same command again the next day.
- **Reruns skip finished runs.** Runs already in `results/results.jsonl` are skipped, so an interrupted batch resumes where it stopped.
- **`--retry-failed` replaces, never duplicates.** It drops each failed row before running it again.
- **No mixed settings.** A results file holding runs for the same model made with other settings or prompt versions is refused.

## Model calls, cost and the $20 cap (`llm.py`)

- **One wrapper for every call.** `call_model` sends Gemini models to the Gemini API and `openai/gpt-oss-20b` to Groq, and logs every call (tokens, cost, latency, error) to `results/runs.jsonl`.
- **Retries.** Up to 5 attempts on HTTP 429, 500 and 503 and on network errors, waiting 5, 10, 20 and 40 s between them; a per-day quota is not retried. `latency_s` includes these waits, so latency partly reflects provider load on the day a condition ran.
- **Prices are required.** A model without a `PRICING` entry is refused, so no cost is ever logged as zero by accident. Free-tier calls, including Groq's, are priced at list price.
- **The cap.** A paid call is refused before it is sent if logged spend plus a small headroom would pass `ADVISORY_BUDGET_USD` (default 20). Spend is summed from the local `results/runs.jsonl`, so the cap holds per machine: all paid study runs went through one machine and key.
- **Groq.** The system prompt is moved into the first user turn, as Groq advises for its reasoning models.

## Seeded reviews (`run_seeded.py`)

Each verified condition's verifier reviews every answer in `data/injected/` (17 with one planted error, 14 correct controls) once, with no revision, through the same agents, calculator and gateway. Each review is saved to `results/seeded/<run_id>.json`. `--dry-run` lists what would run and estimates the cost without calling a model.

## Study data

- `results/` is gitignored on `main`; after each batch it was backed up to the results-only `study-results` branch.
- `scripts/check_study_data.py` is read-only. It confirms that every task × condition × repeat appears exactly once (576 runs), that all conditions share the same first draft and settings, and that the 155 seeded reviews are present.
- `scripts/regrade_results.py` re-grades the saved answers into `results/results_regraded.jsonl` and never edits `results/results.jsonl` or `results/answers/`.

## What happened in the study

- 576 of 576 main runs and 155 of 155 seeded reviews (Nov 19–23), $12.22. The follow-up on 14 new tasks added 168 runs and $3.20.
- 1 failed run: `cross_family` on T06-v003, repeat 2. Groq could not parse the reply, and the retry failed the same way.
- Before the paid runs, a dry run sent the 32 tasks through all six conditions with a fake model; all 192 runs ended in the expected status.

## Tests

The pipeline tests replace every model call with fakes, so they need no network, key or budget:

| File | What it checks |
|---|---|
| `tests/test_pipeline.py` | The `none` condition, saved drafts and the run-and-grade step |
| `tests/test_self_peer.py` | `self` and `peer`: verify, revise once after a reject, verify again |
| `tests/test_dedicated_conditions.py` | `dedicated`, `weakened` and `cross_family` in the same verify-revise-verify flow |
| `tests/test_dedicated.py`, `tests/test_calculator.py` | The dedicated verifier's prompt and calculator loop, and the calculator whitelist |
| `tests/test_failed_runs.py` | API errors saved as failed runs; the daily quota and code errors saved as nothing |
| `tests/test_reruns.py` | Usage counted per attempt, failed runs retried, mixed settings refused |
| `tests/test_llm.py`, `tests/test_groq.py` | The model wrapper: logging, retries, pricing, the budget cap and the Groq provider |
| `tests/test_study_check.py` | The completeness check |
