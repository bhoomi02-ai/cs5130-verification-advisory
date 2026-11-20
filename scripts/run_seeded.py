"""Have one condition's verifier review the seeded answers in data/injected/.

Real API calls. Example (dry check of one answer first):

    python scripts/run_seeded.py --condition dedicated --model gemini-3.6-flash --answers S04_arithmetic_ignore_entry_fee

Each review is saved to results/seeded/<run_id>.json (see advisory/seeded.py).
Reviews already saved are skipped, so the same command continues after a daily
limit. A saved review made with other settings stops the script. A failed API
call is saved with status "failed"; add --retry-failed to run those again.
After 3 failed reviews in a row the script stops, since that points to an
outage. Use the same --model and settings as the main study.

Add --dry-run to see what would run, and a cost estimate, without calling
any model:

    python scripts/run_seeded.py --condition peer --model gemini-3.6-flash --dry-run

The estimate is the mean cost of this condition's reviews already saved, so
there is none before the first review.
"""

from __future__ import annotations

import argparse
import sys

from advisory.conditions import VERIFIED
from advisory.drafts import Settings
from advisory import seeded
from advisory.seeded import (
    all_answer_ids,
    check_same_settings,
    run_seeded_one,
    saved_record,
    seeded_run_id,
)

MAX_FAILED_IN_A_ROW = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True, choices=sorted(VERIFIED))
    parser.add_argument("--model", required=True, help="the advisor's exact model name; cross_family's verifier is set in conditions.py")
    parser.add_argument("--answers", nargs="+", help="answer IDs (default: every file in data/injected)")
    parser.add_argument("--repeats", type=int, default=1, help="reviews per answer")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--retry-failed", action="store_true", help="run the reviews saved as failed again")
    parser.add_argument("--dry-run", action="store_true", help="list what would run and estimate the cost; call no model")
    args = parser.parse_args(argv)

    settings = Settings(args.model, args.temperature, args.top_p, args.thinking_level)
    failed_in_a_row = 0
    to_run = []
    for answer_id in args.answers or all_answer_ids():
        for run_number in range(1, args.repeats + 1):
            run_id = seeded_run_id(args.condition, args.model, answer_id, run_number)
            saved = saved_record(run_id)
            if saved is not None:
                try:
                    check_same_settings(saved, args.condition, settings)
                except ValueError as e:
                    print(f"stop  {e}")
                    return 1
                if not (args.retry_failed and saved["status"] == "failed"):
                    print(f"skip  {answer_id} r{run_number}: already saved")
                    continue
            if args.dry_run:
                to_run.append(run_id)
                print(f"plan  {answer_id} r{run_number}")
                continue
            try:
                rec = run_seeded_one(args.condition, answer_id, run_number, settings)
            except Exception as e:  # daily limit or a code/setup error: nothing saved for this review
                print(f"stop  {answer_id} r{run_number}: {type(e).__name__}: {e}")
                return 1
            if rec["status"] == "failed":
                failed_in_a_row += 1
                print(f"fail  {answer_id} r{run_number}: {rec['error']} (saved as failed)")
                if failed_in_a_row >= MAX_FAILED_IN_A_ROW:
                    print(f"stop  {failed_in_a_row} failed reviews in a row; check the API before running again")
                    return 1
                continue
            failed_in_a_row = 0
            kind = "seeded error" if rec["seeded_error"] else "control"
            print(
                f"done  {answer_id} r{run_number} {args.condition} ({kind}): verdict {rec['verdict'] or 'unreadable'}, "
                f"{rec['verifier_calls']} calls, ${rec['cost_usd']:.4f}"
            )
    if args.dry_run:
        print(_estimate(args.condition, args.model, len(to_run)))
    return 0


def _estimate(condition: str, model: str, n_reviews: int) -> str:
    """One line: how many reviews would run and, if any are saved, their likely cost."""
    costs = []
    out_dir = seeded.SEEDED_RESULTS_DIR  # read at call time, so tests can point it elsewhere
    for record in sorted(out_dir.glob(f"seeded-{condition}-{model}-*.json")) if out_dir.exists() else []:
        rec = saved_record(record.stem)
        if rec and rec["status"] == "reviewed" and rec.get("cost_usd") is not None:
            costs.append(rec["cost_usd"])
    line = f"dry run: {n_reviews} review(s) would run, no model was called"
    if not n_reviews:
        return line
    if not costs:
        return line + f"; no saved {condition} reviews to estimate the cost from"
    mean = sum(costs) / len(costs)
    return line + f"; about ${mean * n_reviews:.2f} at ${mean:.4f} per review (mean of {len(costs)} saved)"


if __name__ == "__main__":
    sys.exit(main())
