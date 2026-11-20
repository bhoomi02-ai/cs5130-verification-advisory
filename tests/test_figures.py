"""Result figures written by advisory.report (new charts only)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from advisory.report import analyze


FIGURE_NAMES = (
    "comparisons.png",
    "outcomes.png",
    "catch_outcomes.png",
    "verifier_rates.png",
    "cost_vs_success.png",
)


def _row(**overrides) -> dict:
    row = {
        "run_id": "none-m-S01-r1",
        "draft_run_id": "draft-m-S01-r1",
        "task_id": "S01",
        "condition": "none",
        "run_number": 1,
        "model": "gemini-3.6-flash",
        "temperature": 0.7,
        "top_p": 0.9,
        "thinking_level": None,
        "prompt_version": "advisor-v1",
        "initial_correct": True,
        "final_correct": True,
        "initial_format_failure": False,
        "final_format_failure": False,
        "verdict": None,
        "status": "unchecked",
        "tokens_in": 10,
        "tokens_out": 5,
        "cost_usd": 0.01,
        "latency_s": 1.0,
        "n_calls": 1,
        "n_failed_calls": 0,
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows: list) -> Path:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _six_condition_rows() -> list[dict]:
    """Small fake study: all six conditions, shared drafts, one missing peer pair."""
    rows = []
    # Shared drafts on t1/t2 for none and most verifiers.
    for task_id, initial, final_none in (
        ("t1", False, False),
        ("t2", True, True),
    ):
        rows.append(
            _row(
                task_id=task_id,
                condition="none",
                run_number=1,
                initial_correct=initial,
                final_correct=final_none,
                verdict=None,
                status="unchecked",
                cost_usd=0.01,
            )
        )
    # self: fixes t1
    rows.append(
        _row(
            task_id="t1",
            condition="self",
            run_number=1,
            initial_correct=False,
            final_correct=True,
            verdict="reject",
            status="revised",
            cost_usd=0.03,
        )
    )
    rows.append(
        _row(
            task_id="t2",
            condition="self",
            run_number=1,
            initial_correct=True,
            final_correct=True,
            verdict="accept",
            status="accepted",
            cost_usd=0.03,
        )
    )
    # peer: only t1; t2 absent → still comparable on t1; also a failed-only side
    # for a third task so one comparison note can appear when paired with a
    # condition that has only failed matches on another axis. Keep peer simple.
    rows.append(
        _row(
            task_id="t1",
            condition="peer",
            run_number=1,
            initial_correct=False,
            final_correct=False,
            verdict="accept",
            status="accepted",
            cost_usd=0.025,
        )
    )
    rows.append(
        _row(
            task_id="t2",
            condition="peer",
            run_number=1,
            initial_correct=True,
            final_correct=True,
            verdict="accept",
            status="accepted",
            cost_usd=0.025,
        )
    )
    # dedicated: fixes t1
    rows.append(
        _row(
            task_id="t1",
            condition="dedicated",
            run_number=1,
            initial_correct=False,
            final_correct=True,
            verdict="reject",
            status="revised",
            cost_usd=0.05,
        )
    )
    rows.append(
        _row(
            task_id="t2",
            condition="dedicated",
            run_number=1,
            initial_correct=True,
            final_correct=True,
            verdict="accept",
            status="accepted",
            cost_usd=0.05,
        )
    )
    # weakened
    rows.append(
        _row(
            task_id="t1",
            condition="weakened",
            run_number=1,
            initial_correct=False,
            final_correct=False,
            verdict="abstain",
            status="human_review",
            cost_usd=0.04,
        )
    )
    rows.append(
        _row(
            task_id="t2",
            condition="weakened",
            run_number=1,
            initial_correct=True,
            final_correct=True,
            verdict="accept",
            status="accepted",
            cost_usd=0.04,
        )
    )
    # cross_family
    rows.append(
        _row(
            task_id="t1",
            condition="cross_family",
            run_number=1,
            initial_correct=False,
            final_correct=True,
            verdict="reject",
            status="revised",
            cost_usd=0.02,
        )
    )
    rows.append(
        _row(
            task_id="t2",
            condition="cross_family",
            run_number=1,
            initial_correct=True,
            final_correct=False,
            verdict="reject",
            status="human_review",
            cost_usd=0.02,
        )
    )
    # Extra condition with only failed runs matched against none on t3 → noted pair.
    rows.append(
        _row(
            task_id="t3",
            condition="none",
            run_number=1,
            initial_correct=False,
            final_correct=False,
            verdict=None,
            status="unchecked",
            cost_usd=0.01,
        )
    )
    # Use a seventh label? No — user asked for six. Make peer have an all-failed
    # alternate by adding failed peer rows on t3 only... but then peer still has
    # usable t1/t2. Add a condition that is only failed: we already have six.
    # Instead add failed-only rows for a duplicate path: weakened-style.
    # For "condition with no usable pairs": include only-failed matches for
    # a condition that has no non-failed overlap with none.
    # Drop t3 none; add condition-like by making one of six have only failed.
    # Simpler: keep six conditions; separately test empty-pair via a tiny set.
    return [row for row in rows if row["task_id"] != "t3"]


def test_figures_created_for_all_six_conditions(tmp_path):
    path = _write_jsonl(tmp_path / "results.jsonl", _six_condition_rows())
    out_dir = tmp_path / "report"
    outcome = analyze(results_path=path, out_dir=out_dir, seed=0)
    assert set(outcome["report"]["summary"]["condition"]) == {
        "none",
        "self",
        "peer",
        "dedicated",
        "weakened",
        "cross_family",
    }
    written = []
    for name in FIGURE_NAMES:
        figure = out_dir / name
        assert figure.exists(), name
        assert figure.stat().st_size > 0, name
        written.append(name)
    assert "right_reason.png" not in outcome["paths"]
    assert not (out_dir / "right_reason.png").exists()
    print("figures:", ", ".join(written))


def test_missing_conditions_and_unusable_pair_do_not_crash(tmp_path):
    # Only none and dedicated; dedicated's matched run is failed → noted pair.
    rows = [
        _row(
            task_id="t1",
            condition="none",
            run_number=1,
            initial_correct=False,
            final_correct=False,
            status="unchecked",
            cost_usd=0.01,
        ),
        _row(
            task_id="t1",
            condition="dedicated",
            run_number=1,
            initial_correct=False,
            final_correct=False,
            verdict=None,
            status="failed",
            cost_usd=0.05,
        ),
        # A second dedicated scored run on another task so dedicated still has
        # a summary row with rates, while the none–dedicated pair on t1 is noted.
        _row(
            task_id="t2",
            condition="dedicated",
            run_number=1,
            initial_correct=True,
            final_correct=True,
            verdict="accept",
            status="accepted",
            cost_usd=0.05,
        ),
    ]
    path = _write_jsonl(tmp_path / "results.jsonl", rows)
    out_dir = tmp_path / "report"
    outcome = analyze(results_path=path, out_dir=out_dir, seed=0)
    comparisons = outcome["report"]["comparisons"]
    assert list(comparisons["condition_b"]) == ["dedicated"]
    assert int(comparisons.iloc[0]["n_pairs"]) == 0
    assert (out_dir / "comparisons.png").exists()
    assert (out_dir / "comparisons.png").stat().st_size > 0
    assert (out_dir / "outcomes.png").exists()
    # Verifier plot still writes for dedicated; none has no verifier rates.
    assert (out_dir / "verifier_rates.png").exists()


def test_right_reason_figure_only_when_seeded_files_exist(tmp_path):
    rows = _six_condition_rows()
    path = _write_jsonl(tmp_path / "results.jsonl", rows)
    out_dir = tmp_path / "report"
    seeded = tmp_path / "seeded"
    seeded.mkdir()
    outcome = analyze(
        results_path=path,
        out_dir=out_dir,
        seed=0,
        seeded_dir=seeded,
    )
    assert not (out_dir / "right_reason.png").exists()
    assert "right_reason.png" not in outcome["paths"]

    (seeded / "err.json").write_text(
        json.dumps(
            {
                "run_id": "seeded-dedicated-m-S01_policy_violation_sell_aapl-r1",
                "answer_id": "S01_policy_violation_sell_aapl",
                "scenario_id": "S01",
                "error_type": "policy_violation",
                "seeded_error": True,
                "condition": "dedicated",
                "run_number": 1,
                "model": "gemini-3.6-flash",
                "verifier_model": "gemini-3.6-flash",
                "status": "reviewed",
                "verdict": "reject",
                "reviews": [
                    {
                        "stage": "first",
                        "verdict": "reject",
                        "issues": ["AAPL may not be sold"],
                        "reason": "sells AAPL against no-sell",
                        "calls": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (seeded / "ctrl.json").write_text(
        json.dumps(
            {
                "run_id": "seeded-dedicated-m-S01_control-r1",
                "answer_id": "S01_control",
                "scenario_id": "S01",
                "error_type": "none",
                "seeded_error": False,
                "condition": "dedicated",
                "run_number": 1,
                "model": "gemini-3.6-flash",
                "verifier_model": "gemini-3.6-flash",
                "status": "reviewed",
                "verdict": "accept",
                "reviews": [
                    {
                        "stage": "first",
                        "verdict": "accept",
                        "issues": [],
                        "reason": "looks fine",
                        "calls": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    outcome = analyze(
        results_path=path,
        out_dir=out_dir,
        seed=0,
        seeded_dir=seeded,
    )
    assert (out_dir / "right_reason.png").exists()
    assert (out_dir / "right_reason.png").stat().st_size > 0
    assert outcome["paths"]["right_reason.png"].exists()


def test_figures_are_deterministic(tmp_path):
    path = _write_jsonl(tmp_path / "results.jsonl", _six_condition_rows())
    first_dir = tmp_path / "report_a"
    second_dir = tmp_path / "report_b"
    analyze(results_path=path, out_dir=first_dir, seed=0)
    analyze(results_path=path, out_dir=second_dir, seed=0)
    for name in FIGURE_NAMES:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
