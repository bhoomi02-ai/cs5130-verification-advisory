"""Build report tables and a chart from results/results.jsonl.

Examples:
    python scripts/analyze.py
    python scripts/analyze.py --results results/results.jsonl --out results/report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from advisory.report import REPORT_DIR, RESULTS_PATH, analyze


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH, help="pipeline results JSONL")
    parser.add_argument("--out", type=Path, default=REPORT_DIR, help="directory for tables and the chart")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed for task-success CIs")
    args = parser.parse_args(argv)

    try:
        outcome = analyze(results_path=args.results, out_dir=args.out, seed=args.seed)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    n_ok = len(outcome["accepted"])
    n_bad = len(outcome["rejected"])
    print(f"Accepted {n_ok} run(s); rejected {n_bad}.")
    if n_bad:
        for item in outcome["rejected"]:
            print(f"  line {item['line']}: {item['reason']}")
    print(f"Wrote report to {args.out}")
    for name, path in outcome["paths"].items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
