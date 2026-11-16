"""Run one condition over a set of scenarios and save one result row per run.

Real API calls: start small, the free tier has a daily limit. Example:

    python scripts/run_condition.py --condition none --model gemini-3.1-flash-lite --tasks S01 --repeats 1

Runs already in results/results.jsonl are skipped, so after hitting the daily
limit you can run the same command again the next day and it continues. A
results file with runs for the same model made with other settings is refused.

A run whose API call fails (after call_model's retries) is saved with status
"failed" and the script moves on; after 3 failed runs in a row it stops, since
that points to an outage rather than the condition. Add --retry-failed to run
the failed ones again, replacing their rows. The daily quota and errors in the
code stop the script without saving a row for that run.
"""

from __future__ import annotations

import argparse
import sys

from advisory.conditions import CONDITIONS
from advisory.drafts import Settings
from advisory.pipeline import RESULTS_PATH, all_task_ids, done_keys, drop_failed, failed_keys, run_one

MAX_FAILED_IN_A_ROW = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--model", required=True, help="exact model name, e.g. gemini-3.1-flash-lite")
    parser.add_argument("--tasks", nargs="+", help="scenario IDs (default: every file in data/scenarios)")
    parser.add_argument("--repeats", type=int, default=1, help="runs per scenario (the study uses 5)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--retry-failed", action="store_true", help="run the runs saved as failed again, replacing their rows")
    args = parser.parse_args(argv)

    settings = Settings(args.model, args.temperature, args.top_p, args.thinking_level)
    tasks = args.tasks or all_task_ids()
    try:
        done = done_keys(RESULTS_PATH, settings=settings)
    except ValueError as e:
        print(f"stop  {e}")
        return 1
    retry = failed_keys(RESULTS_PATH) if args.retry_failed else set()
    failed_in_a_row = 0

    for task_id in tasks:
        for run_number in range(1, args.repeats + 1):
            key = (task_id, args.condition, run_number, args.model)
            if key in done and key not in retry:
                print(f"skip  {task_id} r{run_number}: already in {RESULTS_PATH.name}")
                continue
            if key in retry:
                drop_failed({key}, RESULTS_PATH)
                print(f"retry {task_id} r{run_number}: replacing the failed row")
            try:
                row = run_one(args.condition, task_id, run_number, settings, results_path=RESULTS_PATH)
            except Exception as e:  # daily quota or a code/setup error: nothing saved for this run
                print(f"stop  {task_id} r{run_number}: {type(e).__name__}: {e}")
                return 1
            if row["status"] == "failed":
                failed_in_a_row += 1
                print(f"fail  {task_id} r{run_number} {args.condition}: {row['error']} (saved as a failed run)")
                if failed_in_a_row >= MAX_FAILED_IN_A_ROW:
                    print(f"stop  {failed_in_a_row} failed runs in a row; check the API before running again")
                    return 1
                continue
            failed_in_a_row = 0
            print(
                f"done  {task_id} r{run_number} {args.condition}: "
                f"first draft {'right' if row['initial_correct'] else 'wrong'}, "
                f"final {'right' if row['final_correct'] else 'wrong'}, "
                f"{row['tokens_in'] + row['tokens_out']} tokens, ${row['cost_usd']:.4f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
