"""Show spending so far against the budget, from results/runs.jsonl.

    python scripts/spend.py
    python scripts/spend.py --log some/other/runs.jsonl

The budget is ADVISORY_BUDGET_USD (default 20), the same cap llm.py enforces.
The log is per machine, so run this on the machine that makes the paid calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from advisory import llm
from advisory.spend import format_summary, load_log, spend_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=llm.RUN_LOG, help="run log to read (default: results/runs.jsonl)")
    args = parser.parse_args(argv)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # picks up ADVISORY_BUDGET_USD if set
    print(format_summary(spend_summary(load_log(args.log)), llm.budget_usd()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
