"""Re-grade saved study answers with the current checker (read-only on answers/).

Reads ``results.jsonl`` and ``results/answers/<run_id>.json``, re-runs
``check()`` on the saved draft and final texts, and writes a new JSONL. Never
modifies ``results.jsonl`` or anything under ``results/answers/``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from advisory.checker import check
from advisory.pipeline import RESULTS_PATH, status_correct

DEFAULT_OUT = RESULTS_PATH.with_name("results_regraded.jsonl")
GRADE_FIELDS = (
    "initial_correct",
    "final_correct",
    "initial_format_failure",
    "final_format_failure",
    "initial_status_correct",
    "final_status_correct",
)


def regrade_results(
    results_path: Path = RESULTS_PATH,
    answers_dir: Path | None = None,
    out_path: Path = DEFAULT_OUT,
) -> dict[str, Any]:
    """Re-grade every row; write ``out_path``; return a summary of changes."""
    results_path = Path(results_path)
    answers_dir = Path(answers_dir) if answers_dir is not None else results_path.parent / "answers"
    out_path = Path(out_path)

    rows = _read_jsonl(results_path)
    rewritten: list[dict] = []
    changes: list[dict] = []

    for row in rows:
        if row.get("status") == "failed":
            rewritten.append(dict(row))
            continue
        run_id = row.get("run_id")
        answer_path = answers_dir / f"{run_id}.json"
        if not answer_path.exists():
            # No saved answers: copy through unchanged (same as a failed run).
            rewritten.append(dict(row))
            continue

        record = json.loads(answer_path.read_text(encoding="utf-8"))
        task_id = row.get("task_id") or record.get("task_id")
        draft_text = record.get("draft")
        final_text = record.get("final")
        # When final equals draft, answers files often still store both; grade each text.
        draft_grade = check(task_id, draft_text if isinstance(draft_text, str) else json.dumps(draft_text))
        if final_text == draft_text or final_text is None:
            final_grade = draft_grade
        else:
            final_grade = check(
                task_id,
                final_text if isinstance(final_text, str) else json.dumps(final_text),
            )

        new_row = dict(row)
        new_row["original_initial_correct"] = row.get("initial_correct")
        new_row["original_final_correct"] = row.get("final_correct")
        new_row["initial_correct"] = bool(draft_grade["correct"])
        new_row["final_correct"] = bool(final_grade["correct"])
        new_row["initial_format_failure"] = bool(draft_grade.get("format_failure", False))
        new_row["final_format_failure"] = bool(final_grade.get("format_failure", False))
        new_row["initial_status_correct"] = status_correct(draft_grade)
        new_row["final_status_correct"] = status_correct(final_grade)

        note_parts = []
        if new_row["initial_correct"] != row.get("initial_correct"):
            note_parts.append(
                f"initial_correct {row.get('initial_correct')!r} -> {new_row['initial_correct']!r}"
            )
        if new_row["final_correct"] != row.get("final_correct"):
            note_parts.append(
                f"final_correct {row.get('final_correct')!r} -> {new_row['final_correct']!r}"
            )
        new_row["regrade_note"] = "; ".join(note_parts) if note_parts else ""
        if note_parts:
            changes.append(
                {
                    "run_id": run_id,
                    "task_id": task_id,
                    "condition": row.get("condition"),
                    "initial": (row.get("initial_correct"), new_row["initial_correct"]),
                    "final": (row.get("final_correct"), new_row["final_correct"]),
                    "note": new_row["regrade_note"],
                }
            )
        rewritten.append(new_row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rewritten:
            handle.write(json.dumps(row, default=str) + "\n")

    return {
        "n_rows": len(rows),
        "n_changed": len(changes),
        "out_path": str(out_path),
        "changes": changes,
        "by_task": _group_counts(changes, "task_id"),
        "by_condition": _group_counts(changes, "condition"),
    }


def format_summary(summary: dict[str, Any]) -> str:
    """Human-readable regrade summary."""
    lines = [
        f"Regraded {summary['n_rows']} row(s); {summary['n_changed']} changed.",
        f"Wrote {summary['out_path']}",
    ]
    if not summary["changes"]:
        lines.append("No grade changes.")
        return "\n".join(lines)
    lines.append("By task: " + ", ".join(f"{k}={v}" for k, v in sorted(summary["by_task"].items())))
    lines.append(
        "By condition: "
        + ", ".join(f"{k}={v}" for k, v in sorted(summary["by_condition"].items()))
    )
    lines.append("Changes:")
    for item in summary["changes"]:
        lines.append(
            f"  {item['run_id']}: initial {item['initial'][0]!r}->{item['initial'][1]!r}, "
            f"final {item['final'][0]!r}->{item['final'][1]!r}"
        )
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _group_counts(changes: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in changes:
        counts[str(item.get(key))] += 1
    return dict(counts)
