"""Read-only main-study data checks (advisory.study_check)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from advisory.conditions import CONDITIONS, CROSS_FAMILY_MODEL, VERIFIED
from advisory.pipeline import ROOT
from advisory.prompts import (
    DEDICATED_PROMPT_VERSION,
    PROMPT_VERSION,
    REVISION_PROMPT_VERSION,
    VERIFIER_PROMPT_VERSION,
    WEAKENED_PROMPT_VERSION,
)
from advisory.study_check import (
    STUDY_ADVISOR_MODEL,
    check_study,
    expected_row_settings,
    format_report,
    study_settings,
)

_spec = importlib.util.spec_from_file_location(
    "check_study_data", ROOT / "scripts" / "check_study_data.py"
)
check_study_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_study_data)


TASKS = ("t1", "t2")
REPEATS = 2
ANSWERS = ("ans_err", "ans_ok")


def _expected_verifier_prompt(condition: str) -> str | None:
    return {
        "none": None,
        "self": VERIFIER_PROMPT_VERSION,
        "peer": VERIFIER_PROMPT_VERSION,
        "dedicated": DEDICATED_PROMPT_VERSION,
        "weakened": WEAKENED_PROMPT_VERSION,
        "cross_family": DEDICATED_PROMPT_VERSION,
    }[condition]


def _row(task_id: str, condition: str, run_number: int, **overrides) -> dict:
    advisor = study_settings()
    expected = expected_row_settings(condition, advisor)
    draft_id = f"draft-{advisor.model}-{task_id}-r{run_number}"
    row = {
        "run_id": f"{condition}-{advisor.model}-{task_id}-r{run_number}",
        "draft_run_id": draft_id,
        "task_id": task_id,
        "condition": condition,
        "run_number": run_number,
        "model": expected["model"],
        "temperature": expected["temperature"],
        "top_p": expected["top_p"],
        "thinking_level": expected["thinking_level"],
        "prompt_version": expected["prompt_version"],
        "revision_prompt_version": expected["revision_prompt_version"],
        "verifier_model": expected["verifier_model"],
        "verifier_thinking_level": expected["verifier_thinking_level"],
        "verifier_prompt_version": expected["verifier_prompt_version"],
        "initial_correct": True,
        "final_correct": True,
        "status": "accepted" if condition != "none" else "unchecked",
        "verdict": "accept" if condition != "none" else None,
        "error": None,
    }
    row.update(overrides)
    return row


def _seeded(answer_id: str, condition: str, run_number: int = 1, **overrides) -> dict:
    advisor = study_settings()
    expected = expected_row_settings(condition, advisor)
    record = {
        "run_id": f"seeded-{condition}-{advisor.model}-{answer_id}-r{run_number}",
        "answer_id": answer_id,
        "scenario_id": "t1",
        "error_type": "policy_violation" if answer_id.endswith("err") else "none",
        "seeded_error": answer_id.endswith("err"),
        "condition": condition,
        "run_number": run_number,
        "model": expected["model"],
        "temperature": expected["temperature"],
        "top_p": expected["top_p"],
        "thinking_level": expected["thinking_level"],
        "verifier_model": expected["verifier_model"],
        "verifier_thinking_level": expected["verifier_thinking_level"],
        "verifier_prompt_version": expected["verifier_prompt_version"],
        "status": "reviewed",
        "verdict": "reject" if answer_id.endswith("err") else "accept",
        "reviews": [
            {
                "stage": "first",
                "verdict": "reject" if answer_id.endswith("err") else "accept",
                "issues": ["x"],
                "reason": "because",
                "calls": 1,
            }
        ],
        "error": None,
    }
    record.update(overrides)
    return record


def _write_complete(tmp_path: Path, *, include_seeded: bool = True) -> dict[str, Path]:
    tasks_file = tmp_path / "tasks.txt"
    tasks_file.write_text("\n".join(TASKS) + "\n", encoding="utf-8")

    results = tmp_path / "results" / "results.jsonl"
    results.parent.mkdir(parents=True)
    rows = [
        _row(task_id, condition, run_number)
        for task_id in TASKS
        for condition in CONDITIONS
        for run_number in range(1, REPEATS + 1)
    ]
    results.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    injected = tmp_path / "injected"
    injected.mkdir()
    for answer_id in ANSWERS:
        (injected / f"{answer_id}.json").write_text(
            json.dumps({"answer_id": answer_id, "error_type": "none", "response": {}}),
            encoding="utf-8",
        )

    seeded_dir = tmp_path / "results" / "seeded"
    if include_seeded:
        seeded_dir.mkdir(parents=True)
        for answer_id in ANSWERS:
            for condition in VERIFIED:
                record = _seeded(answer_id, condition)
                path = seeded_dir / f"{record['run_id']}.json"
                path.write_text(json.dumps(record), encoding="utf-8")
    else:
        seeded_dir.mkdir(parents=True)

    return {
        "results": results,
        "seeded_dir": seeded_dir,
        "tasks_file": tasks_file,
        "injected": injected,
        "results_root": results.parent,
    }


def test_expected_settings_come_from_code_constants():
    assert set(CONDITIONS) == {
        "none",
        "self",
        "peer",
        "dedicated",
        "weakened",
        "cross_family",
    }
    none = expected_row_settings("none")
    assert none["model"] == STUDY_ADVISOR_MODEL
    assert none["temperature"] == 0.7
    assert none["top_p"] == 0.9
    assert none["thinking_level"] is None
    assert none["prompt_version"] == PROMPT_VERSION
    assert none["revision_prompt_version"] is None
    assert none["verifier_model"] is None
    assert none["verifier_prompt_version"] is None

    cross = expected_row_settings("cross_family")
    assert cross["verifier_model"] == CROSS_FAMILY_MODEL
    assert cross["verifier_thinking_level"] == "low"
    assert cross["verifier_prompt_version"] == DEDICATED_PROMPT_VERSION
    assert cross["revision_prompt_version"] == REVISION_PROMPT_VERSION

    for condition in VERIFIED:
        assert expected_row_settings(condition)["verifier_prompt_version"] == (
            _expected_verifier_prompt(condition)
        )


def test_complete_fake_dataset_passes(tmp_path):
    paths = _write_complete(tmp_path)
    report = check_study(
        results_path=paths["results"],
        seeded_dir=paths["seeded_dir"],
        tasks_file=paths["tasks_file"],
        repeats=REPEATS,
        injected_dir=paths["injected"],
    )
    assert report.ok
    assert report.missing_runs == []
    assert report.duplicate_runs == []
    coverage = next(item for item in report.summaries if item.name == "coverage")
    assert coverage.expected == len(TASKS) * len(CONDITIONS) * REPEATS
    assert coverage.found == coverage.expected
    seeded = next(item for item in report.summaries if item.name == "seeded_coverage")
    assert seeded.expected == len(ANSWERS) * len(VERIFIED)
    assert seeded.found == seeded.expected

    code = check_study_data.main(
        [
            "--results",
            str(paths["results"]),
            "--seeded-dir",
            str(paths["seeded_dir"]),
            "--injected-dir",
            str(paths["injected"]),
            "--tasks-file",
            str(paths["tasks_file"]),
            "--repeats",
            str(REPEATS),
        ]
    )
    assert code == 0


def test_missing_run_reported(tmp_path):
    paths = _write_complete(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["results"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if not (row["task_id"] == "t1" and row["condition"] == "peer" and row["run_number"] == 2)]
    paths["results"].write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = check_study(
        results_path=paths["results"],
        seeded_dir=paths["seeded_dir"],
        tasks_file=paths["tasks_file"],
        repeats=REPEATS,
        injected_dir=paths["injected"],
    )
    assert not report.ok
    assert any(
        item["task_id"] == "t1" and item["condition"] == "peer" and item["run_number"] == 2
        for item in report.missing_runs
    )
    assert check_study_data.main(
        [
            "--results",
            str(paths["results"]),
            "--seeded-dir",
            str(paths["seeded_dir"]),
            "--injected-dir",
            str(paths["injected"]),
            "--tasks-file",
            str(paths["tasks_file"]),
            "--repeats",
            str(REPEATS),
        ]
    ) == 1


def test_duplicate_run_reported(tmp_path):
    paths = _write_complete(tmp_path)
    with paths["results"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_row("t1", "self", 1)) + "\n")

    report = check_study(
        results_path=paths["results"],
        seeded_dir=paths["seeded_dir"],
        tasks_file=paths["tasks_file"],
        repeats=REPEATS,
        injected_dir=paths["injected"],
    )
    assert not report.ok
    assert any(
        item["task_id"] == "t1" and item["condition"] == "self" and item["count"] == 2
        for item in report.duplicate_runs
    )


def test_duplicate_seeded_reported(tmp_path):
    paths = _write_complete(tmp_path)
    record = _seeded("ans_err", "dedicated")
    (paths["seeded_dir"] / "extra-dup.json").write_text(json.dumps(record), encoding="utf-8")

    report = check_study(
        results_path=paths["results"],
        seeded_dir=paths["seeded_dir"],
        tasks_file=paths["tasks_file"],
        repeats=REPEATS,
        injected_dir=paths["injected"],
    )
    assert not report.ok
    assert any(
        item["answer_id"] == "ans_err"
        and item["condition"] == "dedicated"
        and item["count"] == 2
        for item in report.duplicate_seeded
    )


def test_draft_mismatch_reported(tmp_path):
    paths = _write_complete(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["results"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if row["task_id"] == "t1" and row["run_number"] == 1 and row["condition"] == "peer":
            row["initial_correct"] = False
            row["draft_run_id"] = "draft-other-t1-r1"
    paths["results"].write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = check_study(
        results_path=paths["results"],
        seeded_dir=paths["seeded_dir"],
        tasks_file=paths["tasks_file"],
        repeats=REPEATS,
        injected_dir=paths["injected"],
    )
    assert not report.ok
    match = next(
        item
        for item in report.draft_mismatches
        if item["task_id"] == "t1" and item["run_number"] == 1
    )
    assert "initial_correct" in match["fields"] and "draft_run_id" in match["fields"]
    assert match["disagreeing"] == ["peer"]
    assert "peer: draft-other-t1-r1 (initial_correct=False) [differs]" in match["by_condition"]
    assert "none: draft-gemini-3.6-flash-t1-r1 (initial_correct=True)" in match["by_condition"]
    assert "dedicated: draft-gemini-3.6-flash-t1-r1 (initial_correct=True)" in match["by_condition"]
    # Majority conditions are not marked [differs].
    assert "[differs]" not in match["by_condition"].split("peer:")[0]


def test_settings_mismatch_cross_family_verifier(tmp_path):
    paths = _write_complete(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["results"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if row["condition"] == "cross_family":
            row["verifier_model"] = STUDY_ADVISOR_MODEL  # should be CROSS_FAMILY_MODEL
    paths["results"].write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = check_study(
        results_path=paths["results"],
        seeded_dir=paths["seeded_dir"],
        tasks_file=paths["tasks_file"],
        repeats=REPEATS,
        injected_dir=paths["injected"],
    )
    assert not report.ok
    assert any(
        item["condition"] == "cross_family"
        and item["field"] == "verifier_model"
        and item["expected"] == CROSS_FAMILY_MODEL
        for item in report.settings_mismatches
    )


def test_extra_rows_are_warnings_only(tmp_path):
    paths = _write_complete(tmp_path)
    with paths["results"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_row("S01", "none", 1)) + "\n")  # not in fake task list
        handle.write(json.dumps(_row("t1", "none", 5)) + "\n")  # above repeats

    report = check_study(
        results_path=paths["results"],
        seeded_dir=paths["seeded_dir"],
        tasks_file=paths["tasks_file"],
        repeats=REPEATS,
        injected_dir=paths["injected"],
    )
    assert report.ok
    assert len(report.extra_rows) == 2
    reasons = {tuple(item["reasons"]) for item in report.extra_rows}
    assert ("task_not_in_study",) in reasons
    assert ("run_number_above_repeats",) in reasons


def test_failed_runs_are_warnings_only(tmp_path):
    paths = _write_complete(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["results"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if row["task_id"] == "t2" and row["condition"] == "dedicated" and row["run_number"] == 1:
            row["status"] = "failed"
            row["error"] = "APIError: boom"
    paths["results"].write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = check_study(
        results_path=paths["results"],
        seeded_dir=paths["seeded_dir"],
        tasks_file=paths["tasks_file"],
        repeats=REPEATS,
        injected_dir=paths["injected"],
    )
    assert report.ok
    assert report.failed_by_condition == {"dedicated": 1}
    assert len(report.failed_runs) == 1


def test_empty_seeded_dir_skipped_without_error(tmp_path):
    paths = _write_complete(tmp_path, include_seeded=False)
    report = check_study(
        results_path=paths["results"],
        seeded_dir=paths["seeded_dir"],
        tasks_file=paths["tasks_file"],
        repeats=REPEATS,
        injected_dir=paths["injected"],
    )
    assert report.ok
    assert report.seeded_skipped
    assert "no .json" in report.seeded_skip_reason


def test_script_never_writes_inside_results(tmp_path):
    paths = _write_complete(tmp_path)
    before = {
        path.relative_to(paths["results_root"]): path.read_bytes()
        for path in paths["results_root"].rglob("*")
        if path.is_file()
    }
    json_out = tmp_path / "outside" / "findings.json"
    code = check_study_data.main(
        [
            "--results",
            str(paths["results"]),
            "--seeded-dir",
            str(paths["seeded_dir"]),
            "--injected-dir",
            str(paths["injected"]),
            "--tasks-file",
            str(paths["tasks_file"]),
            "--repeats",
            str(REPEATS),
            "--json",
            str(json_out),
        ]
    )
    assert code == 0
    assert json_out.exists()
    after = {
        path.relative_to(paths["results_root"]): path.read_bytes()
        for path in paths["results_root"].rglob("*")
        if path.is_file()
    }
    assert after == before

    refused = check_study_data.main(
        [
            "--results",
            str(paths["results"]),
            "--seeded-dir",
            str(paths["seeded_dir"]),
            "--injected-dir",
            str(paths["injected"]),
            "--tasks-file",
            str(paths["tasks_file"]),
            "--repeats",
            str(REPEATS),
            "--json",
            str(paths["results_root"] / "leak.json"),
        ]
    )
    assert refused == 1
    assert not (paths["results_root"] / "leak.json").exists()


def test_format_report_mentions_problems(tmp_path):
    paths = _write_complete(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["results"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = rows[:-1]
    paths["results"].write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = check_study(
        results_path=paths["results"],
        seeded_dir=paths["seeded_dir"],
        tasks_file=paths["tasks_file"],
        repeats=REPEATS,
        injected_dir=paths["injected"],
    )
    text = format_report(report)
    assert "FAILED" in text
    assert "Missing study runs" in text
