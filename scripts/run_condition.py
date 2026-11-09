"""Run one condition over a set of scenarios and save one result row per run.

Real API calls: start small, the free tier has a daily limit. Example:

    python scripts/run_condition.py --condition none --model gemini-3.1-flash-lite --tasks S01 --repeats 1

Runs already in results/results.jsonl are skipped, so after hitting the daily
limit you can run the same command again the next day and it continues.
"""

from __future__ import annotations

import argparse
import sys

from advisory.conditions import CONDITIONS
from advisory.drafts import Settings
from advisory.pipeline import RESULTS_PATH, all_task_ids, done_keys, run_one


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--model", required=True, help="exact model name, e.g. gemini-3.1-flash-lite")
    parser.add_argument("--tasks", nargs="+", help="scenario IDs (default: every file in data/scenarios)")
    parser.add_argument("--repeats", type=int, default=1, help="runs per scenario (the study uses 5)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high"])
    args = parser.parse_args(argv)

    settings = Settings(args.model, args.temperature, args.top_p, args.thinking_level)
    tasks = args.tasks or all_task_ids()
    done = done_keys()

    for task_id in tasks:
        for run_number in range(1, args.repeats + 1):
            if (task_id, args.condition, run_number, args.model) in done:
                print(f"skip  {task_id} r{run_number}: already in {RESULTS_PATH.name}")
                continue
            try:
                row = run_one(args.condition, task_id, run_number, settings)
            except Exception as e:  # e.g. the daily quota; finished runs are already saved
                print(f"stop  {task_id} r{run_number}: {type(e).__name__}: {e}")
                return 1
            print(
                f"done  {task_id} r{run_number} {args.condition}: "
                f"first draft {'right' if row['initial_correct'] else 'wrong'}, "
                f"final {'right' if row['final_correct'] else 'wrong'}, "
                f"{row['tokens_in'] + row['tokens_out']} tokens, ${row['cost_usd']:.4f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
