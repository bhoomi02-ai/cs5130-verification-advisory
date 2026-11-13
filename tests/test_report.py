"""Report builder tests. No network; results files are hand-made JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from advisory.report import analyze, build_report, load_results


def _row(**overrides) -> dict:
    row = {
        "run_id": "none-m-S01-r1",
        "draft_run_id": "draft-m-S01-r1",
        "task_id": "S01",
        "condition": "none",
        "run_number": 1,
        "model": "gemini-3.1-flash-lite",
        "temperature": 0.7,
        "top_p": 0.9,
        "thinking_level": None,
        "prompt_version": "advisor-v0",
        "initial_correct": True,
        "final_correct": True,
        "initial_format_failure": False,
        "final_format_failure": False,
        "verdict": None,
        "status": "unchecked",
        "tokens_in": 10,
        "tokens_out": 5,
        "cost_usd": 0.001,
        "latency_s": 1.0,
        "n_calls": 1,
        "n_failed_calls": 0,
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows: list) -> Path:
    lines = []
    for item in rows:
        if isinstance(item, str):
            lines.append(item)
        else:
            lines.append(json.dumps(item))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_normal_results_file(tmp_path):
    path = _write_jsonl(
        tmp_path / "results.jsonl",
        [
            _row(task_id="S01", run_number=1, final_correct=True, initial_correct=True, cost_usd=3),
            _row(task_id="S02", run_number=1, final_correct=False, initial_correct=True, cost_usd=3),
            _row(
                task_id="S01",
                condition="dedicated",
                run_number=1,
                final_correct=True,
                initial_correct=False,
                verdict="reject",
                status="revised",
                cost_usd=9,
            ),
            _row(
                task_id="S09",
                condition="none",
                run_number=1,
                final_correct=False,
                initial_correct=False,
                final_status_correct=True,
                cost_usd=1,
            ),
            _row(
                task_id="S10",
                condition="none",
                run_number=1,
                final_correct=False,
                initial_correct=False,
                final_status_correct=False,
                cost_usd=1,
            ),
        ],
    )
    outcome = analyze(results_path=path, out_dir=tmp_path / "report", seed=0)
    assert outcome["rejected"] == []
    summary = outcome["report"]["summary"]
    assert set(summary["condition"]) == {"none", "dedicated"}
    assert (tmp_path / "report" / "summary.csv").exists()
    assert (tmp_path / "report" / "summary.md").exists()
    assert (tmp_path / "report" / "by_scenario.csv").exists()
    assert (tmp_path / "report" / "task_success.png").exists()
    assert (tmp_path / "report" / "task_success.png").stat().st_size > 0

    by_scenario = pd.read_csv(tmp_path / "report" / "by_scenario.csv")
    s01_none = by_scenario[(by_scenario.task_id == "S01") & (by_scenario.condition == "none")]
    assert float(s01_none.success_rate.iloc[0]) == 1.0

    infeasible = pd.read_csv(tmp_path / "report" / "infeasible_status.csv")
    s09 = infeasible[(infeasible.task_id == "S09") & (infeasible.condition == "none")].iloc[0]
    s10 = infeasible[(infeasible.task_id == "S10") & (infeasible.condition == "none")].iloc[0]
    assert s09.strict_success == 0.0 and s09.status_only_success == 1.0
    assert s10.strict_success == 0.0 and s10.status_only_success == 0.0


def test_two_models(tmp_path):
    path = _write_jsonl(
        tmp_path / "results.jsonl",
        [
            _row(model="a", task_id="S01", final_correct=True, cost_usd=3),
            _row(model="a", task_id="S02", final_correct=False, cost_usd=3),
            _row(
                model="a",
                condition="dedicated",
                task_id="S01",
                final_correct=True,
                verdict="accept",
                status="accepted",
                cost_usd=9,
            ),
            _row(model="b", task_id="S01", final_correct=False, cost_usd=1),
            _row(
                model="b",
                condition="dedicated",
                task_id="S01",
                final_correct=True,
                initial_correct=False,
                verdict="reject",
                status="revised",
                cost_usd=2,
            ),
        ],
    )
    accepted, rejected = load_results(path)
    assert rejected == []
    summary = build_report(accepted, seed=0)["summary"]
    pairs = set(zip(summary["condition"], summary["model"]))
    assert pairs == {("none", "a"), ("dedicated", "a"), ("none", "b"), ("dedicated", "b")}
    dedicated_a = summary[(summary.condition == "dedicated") & (summary.model == "a")].iloc[0]
    dedicated_b = summary[(summary.condition == "dedicated") & (summary.model == "b")].iloc[0]
    assert dedicated_a.extra_cost_per_extra_correct == 12
    assert dedicated_b.extra_cost_per_extra_correct == 1


def test_failed_run(tmp_path):
    path = _write_jsonl(
        tmp_path / "results.jsonl",
        [
            _row(
                condition="dedicated",
                task_id="S01",
                run_number=1,
                initial_correct=False,
                final_correct=False,
                verdict="reject",
                status="revised",
            ),
            _row(
                condition="dedicated",
                task_id="S02",
                run_number=1,
                initial_correct=True,
                final_correct=True,
                verdict="accept",
                status="accepted",
            ),
            _row(
                condition="dedicated",
                task_id="S03",
                run_number=1,
                initial_correct=True,
                final_correct=True,
                verdict=None,
                status="failed",
                cost_usd=99,
            ),
        ],
    )
    outcome = analyze(results_path=path, out_dir=tmp_path / "report")
    assert outcome["rejected"] == []
    summary = outcome["report"]["summary"]
    row = summary[summary.condition == "dedicated"].iloc[0]
    assert row.n_failed == 1
    assert row.task_success == 0.5
    by_scenario = outcome["report"]["by_scenario"]
    assert set(by_scenario.task_id) == {"S01", "S02"}


def test_malformed_row_is_reported_not_dropped(tmp_path):
    path = _write_jsonl(
        tmp_path / "results.jsonl",
        [
            _row(task_id="S01", final_correct=True),
            "{not json",
            _row(task_id="S02", final_correct=False),
            {"task_id": "S03", "condition": "none"},  # missing required columns
        ],
    )
    accepted, rejected = load_results(path)
    assert len(accepted) == 2
    assert len(rejected) == 2
    reasons = {item["reason"] for item in rejected}
    assert any(reason.startswith("invalid JSON") for reason in reasons)
    assert any(reason.startswith("missing columns") for reason in reasons)

    outcome = analyze(results_path=path, out_dir=tmp_path / "report")
    assert len(outcome["rejected"]) == 2
    rejected_csv = pd.read_csv(tmp_path / "report" / "rejected_rows.csv")
    assert len(rejected_csv) == 2
    rejected_md = (tmp_path / "report" / "rejected.md").read_text(encoding="utf-8")
    assert "2 row(s) were not summarised" in rejected_md


def test_all_failed_condition_still_writes(tmp_path):
    # none has a scored run; self is entirely failed. The report still writes,
    # and the self row shows n_failed with empty rates.
    path = _write_jsonl(
        tmp_path / "results.jsonl",
        [
            _row(
                condition="none",
                task_id="S01",
                run_number=1,
                final_correct=True,
                initial_correct=True,
                status="unchecked",
                cost_usd=1,
            ),
            _row(
                condition="self",
                task_id="S01",
                run_number=1,
                final_correct=False,
                initial_correct=False,
                verdict=None,
                status="failed",
                cost_usd=4,
            ),
            _row(
                condition="self",
                task_id="S02",
                run_number=1,
                final_correct=False,
                initial_correct=False,
                verdict=None,
                status="failed",
                cost_usd=6,
            ),
        ],
    )
    outcome = analyze(results_path=path, out_dir=tmp_path / "report", seed=0)
    assert outcome["rejected"] == []
    assert (tmp_path / "report" / "summary.csv").exists()
    assert (tmp_path / "report" / "task_success.png").stat().st_size > 0
    summary = outcome["report"]["summary"]
    self_row = summary[summary.condition == "self"].iloc[0]
    assert self_row.n_failed == 2
    assert self_row.task_success is None or pd.isna(self_row.task_success)
    assert self_row.catch_rate is None or pd.isna(self_row.catch_rate)
    assert self_row.mean_cost_usd is None or pd.isna(self_row.mean_cost_usd)
    assert self_row.extra_cost_per_extra_correct is None or pd.isna(
        self_row.extra_cost_per_extra_correct
    )
    none_row = summary[summary.condition == "none"].iloc[0]
    assert none_row.task_success == 1.0
