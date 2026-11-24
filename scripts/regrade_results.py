"""Re-grade saved answers with the current checker into a new JSONL.

Never modifies results/results.jsonl or results/answers/. Example:

    python scripts/regrade_results.py
    python scripts/regrade_results.py --results results/results.jsonl \\
        --out results/results_regraded.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from advisory.pipeline import RESULTS_PATH
from advisory.regrade import DEFAULT_OUT, format_summary, regrade_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help="input results JSONL (read-only)",
    )
    parser.add_argument(
        "--answers-dir",
        type=Path,
        default=None,
        help="answers folder (default: <results-parent>/answers)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="new JSONL path (never the input file)",
    )
    args = parser.parse_args(argv)

    if args.out.resolve() == Path(args.results).resolve():
        print("error: --out must differ from --results", file=sys.stderr)
        return 1
    if not Path(args.results).exists():
        print(f"error: {args.results} not found", file=sys.stderr)
        return 1

    summary = regrade_results(
        results_path=args.results,
        answers_dir=args.answers_dir,
        out_path=args.out,
    )
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
