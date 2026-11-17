"""Spending so far, from the run log (results/runs.jsonl).

Every model call goes through llm.call_model and is logged with its cost, so
the log is the ledger the $20 cap checks against. This module adds it up by
model, condition and day. Free-tier calls are logged at paid prices, so the
totals can only overstate what was actually charged.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_log(path: Path) -> list[dict]:
    """Every record in the run log; a missing file means nothing spent yet."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def spend_summary(records: list[dict]) -> dict:
    """Totals overall and grouped by model, condition and UTC day."""

    def group(key) -> list[dict]:
        rows: dict = defaultdict(lambda: {"calls": 0, "failed": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})
        for r in records:
            row = rows[key(r)]
            row["calls"] += 1
            row["failed"] += 1 if r.get("error") else 0
            row["tokens_in"] += r.get("tokens_in") or 0
            row["tokens_out"] += r.get("tokens_out") or 0
            row["cost_usd"] += r.get("cost_usd") or 0.0
        return sorted(({"name": k, **v} for k, v in rows.items()), key=lambda x: -x["cost_usd"])

    return {
        "calls": len(records),
        "failed": sum(1 for r in records if r.get("error")),
        "cost_usd": sum(r.get("cost_usd") or 0.0 for r in records),
        "by_model": group(lambda r: r.get("model") or "?"),
        "by_condition": group(lambda r: r.get("condition") or "?"),
        "by_day": sorted(group(lambda r: (r.get("timestamp") or "?")[:10]), key=lambda x: x["name"]),
    }


def format_summary(summary: dict, budget: float) -> str:
    """A plain-text report: total against the budget, then one table per grouping."""
    spent = summary["cost_usd"]
    lines = [
        f"Spent ${spent:.2f} of ${budget:.2f} ({spent / budget:.0%}); ${max(budget - spent, 0):.2f} left"
        if budget > 0
        else f"Spent ${spent:.2f} (no budget set)",
        f"{summary['calls']} calls logged, {summary['failed']} failed",
        "Free-tier calls are logged at paid prices, so this can only overstate what was charged.",
    ]
    for title, key in (("By model", "by_model"), ("By condition", "by_condition"), ("By day (UTC)", "by_day")):
        lines += ["", title, f"  {'':28}{'calls':>7}{'failed':>8}{'tokens in':>12}{'tokens out':>12}{'cost':>10}"]
        for row in summary[key]:
            lines.append(
                f"  {str(row['name'])[:28]:28}{row['calls']:>7}{row['failed']:>8}"
                f"{row['tokens_in']:>12,}{row['tokens_out']:>12,}{'$%.4f' % row['cost_usd']:>10}"
            )
    return "\n".join(lines)
