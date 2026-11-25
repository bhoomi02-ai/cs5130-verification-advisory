"""Build the web dashboard's data files from the saved results. Read-only; no model calls.

    python dashboard/build_data.py                         # results/ -> dashboard/data/
    python dashboard/build_data.py --results path/to/results --out somewhere/

See dashboard/README.md. The output is gitignored on main; it is derived from
the study results, which live on the study-results branch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from advisory.dashboard_export import OUT_DIR, RESULTS_DIR, STUDY_REPEATS, build
from advisory.pipeline import STUDY_TASKS_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR, help="results folder (default: results/)")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output folder, replaced on each run (default: dashboard/data/)")
    parser.add_argument("--results-file", type=Path, default=None,
                        help="results JSONL to show (default: <results>/results.jsonl); e.g. results/results_regraded.jsonl "
                             "from scripts/regrade_results.py, whose original grades become the sensitivity check")
    parser.add_argument("--tasks-file", type=Path, default=STUDY_TASKS_PATH)
    parser.add_argument("--repeats", type=int, default=STUDY_REPEATS)
    args = parser.parse_args(argv)
    results_file = args.results_file or args.results / "results.jsonl"
    if not results_file.exists():
        print(f"No {results_file.name} in {results_file.parent}", file=sys.stderr)
        return 1
    meta = build(args.results, args.out, args.tasks_file, args.repeats, results_file=results_file)
    done = ", ".join(f"{c['id']} {c['runs_done']}/{c['runs_expected']}" for c in meta["conditions"])
    print(f"Wrote {len(meta['files']) - 1} files and {meta['run_details']} run details to {args.out}")
    print(f"Runs: {done}")
    print(f"Complete: {', '.join(meta['complete_conditions']) or 'none'}")
    if meta["grading"]["regraded"]:
        print(f"Re-graded file: {len(meta['grading']['changed_runs'])} run(s) changed grade; original grading exported as a sensitivity check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
