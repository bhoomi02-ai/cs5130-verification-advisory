"""Run the model pilot (issue #1).

Examples:
    python scripts/run_pilot.py --scenarios S01             # one quick live call
    python scripts/run_pilot.py                             # both models, all scenarios
    python scripts/run_pilot.py --models gemini-2.5-flash --repeats 3

Answers go to results/pilot/<model>/, every model call to results/runs.jsonl.
"""

from __future__ import annotations

import argparse

from advisory.pilot import load_scenarios, run_pilot, summarize

DEFAULT_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--scenarios", nargs="+", help="scenario ids, e.g. S01 S02 (default: all)")
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()

    # TODO: pass grade=<checker function> once checker.py is ready (Dhanya).
    results = run_pilot(args.models, load_scenarios(ids=args.scenarios), repeats=args.repeats)

    print("\nmodel                     runs  schema-valid  valid-1st-try  accuracy")
    for row in summarize(results):
        acc = "n/a (no grader)" if row["accuracy"] is None else f"{row['accuracy']:.0%}"
        print(
            f"{row['model']:24} {row['runs']:5}  {row['schema_valid']:12.0%}"
            f"  {row['valid_first_try']:13.0%}  {acc}"
        )


if __name__ == "__main__":
    main()
