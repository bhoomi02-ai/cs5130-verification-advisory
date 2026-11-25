"""Re-grade tool: new JSONL only; answers/ and results.jsonl untouched."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from advisory.evaluation.reference import reference_response
from advisory.regrade import format_summary, regrade_results
from tests.s06_helpers import S06_FAMILY, s06_family_rounded_only

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "regrade_results", _ROOT / "scripts" / "regrade_results.py"
)
regrade_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(regrade_script)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(run_id: str, task_id: str, condition: str, initial: bool, final: bool, **extra) -> dict:
    row = {
        "run_id": run_id,
        "draft_run_id": f"draft-{run_id}",
        "task_id": task_id,
        "condition": condition,
        "run_number": 1,
        "model": "gemini-3.6-flash",
        "initial_correct": initial,
        "final_correct": final,
        "initial_format_failure": False,
        "final_format_failure": False,
        "initial_status_correct": True,
        "final_status_correct": True,
        "status": "accepted",
        "verdict": "accept",
        "error": None,
    }
    row.update(extra)
    return row


def _answer(run_id: str, task_id: str, condition: str, draft, final=None) -> dict:
    draft_text = draft if isinstance(draft, str) else json.dumps(draft)
    final_obj = draft if final is None else final
    final_text = final_obj if isinstance(final_obj, str) else json.dumps(final_obj)
    return {
        "run_id": run_id,
        "task_id": task_id,
        "condition": condition,
        "run_number": 1,
        "draft": draft_text,
        "revision": None,
        "final": final_text,
        "verdict": "accept",
        "status": "accepted",
    }


def test_regrade_only_s06_family_can_change_and_original_untouched(tmp_path):
    results = tmp_path / "results.jsonl"
    answers = tmp_path / "answers"
    out = tmp_path / "results_regraded.jsonl"
    answers.mkdir()

    # S01: already correct under old and new rules.
    s01 = reference_response("S01")
    # S06: rounded-only answer was wrong under the old rule (missing unrounded),
    # stored as incorrect; new checker marks it correct.
    s06 = s06_family_rounded_only("S06")

    rows = [
        _row("none-m-S01-r1", "S01", "none", True, True),
        _row("dedicated-m-S06-r1", "S06", "dedicated", False, False),
        _row("peer-m-S06-r1", "S06", "peer", False, False, status="failed", error="APIError"),
    ]
    _write_jsonl(results, rows)
    (answers / "none-m-S01-r1.json").write_text(
        json.dumps(_answer("none-m-S01-r1", "S01", "none", s01)), encoding="utf-8"
    )
    (answers / "dedicated-m-S06-r1.json").write_text(
        json.dumps(_answer("dedicated-m-S06-r1", "S06", "dedicated", s06)), encoding="utf-8"
    )
    before = results.read_bytes()
    answer_before = {
        path.name: path.read_bytes() for path in answers.glob("*.json")
    }

    summary = regrade_results(results_path=results, answers_dir=answers, out_path=out)
    assert results.read_bytes() == before
    assert {path.name: path.read_bytes() for path in answers.glob("*.json")} == answer_before

    assert summary["n_rows"] == 3
    assert summary["n_changed"] == 1
    assert summary["by_task"] == {"S06": 1}
    assert summary["by_condition"] == {"dedicated": 1}

    new_rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    by_id = {row["run_id"]: row for row in new_rows}
    assert by_id["none-m-S01-r1"]["initial_correct"] is True
    assert by_id["none-m-S01-r1"]["final_correct"] is True
    assert by_id["none-m-S01-r1"]["original_initial_correct"] is True
    assert by_id["none-m-S01-r1"]["regrade_note"] == ""
    # Failed run with no answers file is copied unchanged.
    assert by_id["peer-m-S06-r1"]["status"] == "failed"
    assert by_id["peer-m-S06-r1"]["initial_correct"] is False
    assert "regrade_note" not in by_id["peer-m-S06-r1"]
    assert "original_initial_correct" not in by_id["peer-m-S06-r1"]

    changed = by_id["dedicated-m-S06-r1"]
    assert changed["original_initial_correct"] is False
    assert changed["original_final_correct"] is False
    assert changed["initial_correct"] is True
    assert changed["final_correct"] is True
    assert "initial_correct" in changed["regrade_note"]

    # Non-S06 family tasks in the fixture did not change.
    assert not any(item["task_id"] not in S06_FAMILY for item in summary["changes"])

    text = format_summary(summary)
    assert "1 changed" in text
    assert "dedicated-m-S06-r1" in text

    code = regrade_script.main(
        ["--results", str(results), "--answers-dir", str(answers), "--out", str(tmp_path / "cli.jsonl")]
    )
    assert code == 0
    assert (tmp_path / "cli.jsonl").exists()


def test_regrade_refuses_to_overwrite_input(tmp_path):
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [_row("x", "S01", "none", True, True)])
    assert regrade_script.main(["--results", str(results), "--out", str(results)]) == 1
