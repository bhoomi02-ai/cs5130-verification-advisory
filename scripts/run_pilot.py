"""Run the model pilot (issue #1).

Examples:
    python scripts/run_pilot.py --dry-run                   # show the plan, make no calls
    python scripts/run_pilot.py --scenarios S01             # one quick live run
    python scripts/run_pilot.py --models gemini-3.6-flash --thinking-level low --repeats 3
    python scripts/run_pilot.py --models gemini-3.6-flash --repeats 3 --resume 20261113T021603Z

If the daily quota stops a batch, the finished runs stay saved; run the same
command the next day with --resume <batch_id> to do only the missing runs.

Each batch is saved to results/pilot/<batch_id>/<model>/, and every model call
is logged to results/runs.jsonl with the same run_id. Answers are graded with
the hidden checker unless --no-grade is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from google.genai import errors as genai_errors

from advisory.llm import THINKING_LEVELS
from advisory.pilot import (
    DEFAULT_SETTINGS,
    PILOT_DIR,
    PROMPT_VERSION,
    checker_grade,
    load_scenarios,
    planned_calls,
    run_pilot,
    summarize,
)

DEFAULT_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--scenarios", nargs="+", help="scenario ids, e.g. S01 S02 (default: all)")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--temperature", type=float, default=DEFAULT_SETTINGS["temperature"])
    p.add_argument("--top-p", type=float, default=DEFAULT_SETTINGS["top_p"])
    p.add_argument("--thinking-level", choices=THINKING_LEVELS, default=DEFAULT_SETTINGS["thinking_level"])
    p.add_argument("--max-calls", type=int, default=100, help="refuse to start above this many calls (default 100)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit without calling any model")
    p.add_argument("--no-grade", action="store_true", help="skip grading with the checker")
    p.add_argument("--resume", metavar="BATCH_ID", help="continue this batch: keep its saved runs, do only the missing ones")
    args = p.parse_args()

    scenarios = load_scenarios(ids=args.scenarios)
    batch_id = args.resume or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    settings = {"temperature": args.temperature, "top_p": args.top_p, "thinking_level": args.thinking_level}
    n = planned_calls(args.models, scenarios, args.repeats)
    print(
        f"Plan: {n} call(s) = {len(args.models)} model(s) x {len(scenarios)} scenario(s) x {args.repeats} repeat(s)\n"
        f"Models: {', '.join(args.models)}\nSettings: {settings}, prompt {PROMPT_VERSION}\n"
        f"Output: {PILOT_DIR / batch_id}/" + (" (resuming: saved runs are not called again)" if args.resume else "")
    )
    if args.dry_run:
        return

    try:
        results = run_pilot(
            args.models,
            scenarios,
            repeats=args.repeats,
            grade=None if args.no_grade else checker_grade,
            settings=settings,
            max_calls=args.max_calls,
            batch_id=batch_id,
            resume=bool(args.resume),
        )
    except genai_errors.APIError as e:  # run_pilot only lets the daily quota through
        print(f"\nStopped: daily quota reached ({type(e).__name__} {e.code}). Finished runs are saved in {PILOT_DIR / batch_id}.")
        print(f"After the quota resets, run the same command with --resume {batch_id}")
        sys.exit(1)
    rows = summarize(results)
    batch_dir = PILOT_DIR / batch_id
    (batch_dir / "summary.json").write_text(
        json.dumps({"prompt_version": PROMPT_VERSION, "settings": settings, "summary": rows}, indent=2)
    )

    print("\nmodel                     runs  api-errors  schema-valid  accuracy")
    for row in rows:
        valid = "n/a" if row["schema_valid"] is None else f"{row['schema_valid']:.0%}"
        acc = "n/a" if row["accuracy"] is None else f"{row['accuracy']:.0%} of {row['graded_runs']}"
        print(f"{row['model']:24} {row['runs']:5}  {row['api_errors']:10}  {valid:>12}  {acc}")
    for row in rows:
        if row["failures"]:
            print(f"\n{row['model']} failed checks:")
            for label, count in row["failures"].items():
                print(f"  {count:3}  {label}")
    print(f"\nSaved to {batch_dir}")


if __name__ == "__main__":
    main()
