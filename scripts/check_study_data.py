"""Read-only check that main-study results and seeded reviews are complete.

Never writes under results/ and never calls a model.

Examples:

    python scripts/check_study_data.py
    python scripts/check_study_data.py --json /tmp/study_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from advisory.pipeline import RESULTS_PATH, STUDY_TASKS_PATH
from advisory.seeded import SEEDED_DIR, SEEDED_RESULTS_DIR
from advisory.study_check import (
    STUDY_ADVISOR_MODEL,
    STUDY_REPEATS,
    assert_json_path_outside_results,
    check_study,
    format_report,
    study_settings,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help="pipeline results JSONL (read-only)",
    )
    parser.add_argument(
        "--seeded-dir",
        type=Path,
        default=SEEDED_RESULTS_DIR,
        help="seeded review JSON directory (read-only)",
    )
    parser.add_argument(
        "--injected-dir",
        type=Path,
        default=SEEDED_DIR,
        help="seeded answer files used for coverage (read-only)",
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=STUDY_TASKS_PATH,
        help="study task list (one ID per line)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=STUDY_REPEATS,
        help="expected run_number values 1..repeats",
    )
    parser.add_argument(
        "--model",
        default=STUDY_ADVISOR_MODEL,
        help="expected advisor model (Settings defaults supply temp/top_p/thinking)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        dest="json_path",
        help="write full findings JSON here (must not be under results/)",
    )
    args = parser.parse_args(argv)

    if args.json_path is not None:
        try:
            assert_json_path_outside_results(args.json_path, args.results)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    report = check_study(
        results_path=args.results,
        seeded_dir=args.seeded_dir,
        tasks_file=args.tasks_file,
        repeats=args.repeats,
        injected_dir=args.injected_dir,
        advisor=study_settings(args.model),
    )
    print(format_report(report), end="")

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote findings to {args.json_path}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
