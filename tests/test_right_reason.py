"""Right-reason scoring for seeded-error reviews (scorer-side only)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from advisory.report import analyze
from advisory.right_reason import (
    audit_table,
    classify_review,
    extract_numbers,
    load_labels,
    right_reason_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "evaluation" / "seeded_error_labels.csv"
_AGENT_MODULES = (
    "prompts.py",
    "conditions.py",
    "dedicated.py",
    "pipeline.py",
    "llm.py",
    "verdict.py",
)


def _label(answer_id: str) -> dict:
    return next(row for row in load_labels(LABELS) if row["answer_id"] == answer_id)


def _review(verdict="reject", issues=None, reason="", **extra) -> dict:
    row = {
        "stage": "first",
        "verdict": verdict,
        "issues": [] if issues is None else issues,
        "reason": reason,
        "error": None,
        "raw": "",
        "trailing": "",
        "calls": 1,
    }
    row.update(extra)
    return row


def _record(answer_id: str, review: dict, **extra) -> dict:
    row = {
        "answer_id": answer_id,
        "scenario_id": answer_id.split("_")[0],
        "condition": "dedicated",
        "run_number": 1,
        "verifier_model": "gemini-test",
        "verifier_prompt_version": "dedicated-v0",
        "reviews": [review],
    }
    row.update(extra)
    return row


def _results_row(**overrides) -> dict:
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


def test_agents_do_not_import_right_reason_or_labels():
    for name in _AGENT_MODULES:
        source = (ROOT / "src" / "advisory" / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [module, *[f"{module}.{alias.name}" for alias in node.names]]
            else:
                continue
            for imported in names:
                assert "right_reason" not in imported, name
                assert "seeded_error_labels" not in imported, name
        assert "seeded_error_labels.csv" not in source, name


def test_load_labels_parses_real_csv():
    labels = load_labels(LABELS)
    assert len(labels) == 17
    s04 = _label("S04_arithmetic_ignore_entry_fee")
    assert s04["wrong_numbers"] == [139079.42]
    assert s04["correct_numbers"] == [137688.62]
    assert s04["also_wrong_numbers"] == [6389.93, 7780.72]
    assert "Fund A" in s04["match_terms"]
    # Non-numeric status / listed / missing fields contribute no numbers.
    status = _label("S03_wrong_status_feasible")
    assert status["wrong_numbers"] == []
    assert status["correct_numbers"] == []
    listed = _label("S03_violation_list_cash_at_limit")
    assert listed["wrong_numbers"] == []
    missing = _label("S03_violation_list_missed_crypto")
    assert missing["wrong_numbers"] == []


def test_extract_numbers_handles_currency_commas_percent_and_sign():
    assert extract_numbers("$137,688.62") == [137688.62]
    assert extract_numbers("137,689") == [137689.0]
    assert extract_numbers("35%") == [35.0, 0.35]
    assert extract_numbers("-1") == [-1.0]


def test_number_matching_tolerance_and_formats():
    label = _label("S04_arithmetic_ignore_entry_fee")
    exact = classify_review(_review(issues=["Fund A ends at 139079.42"]), label)
    assert exact["decision"] == "right_reason"
    comma = classify_review(_review(issues=["Final value 137,688.62"]), label)
    assert comma["decision"] == "right_reason"
    dollars = classify_review(_review(issues=["Correct is $137688.62"]), label)
    assert dollars["decision"] == "right_reason"
    rounded = classify_review(_review(issues=["About 137,689"]), label)
    assert rounded["decision"] == "right_reason"
    far = classify_review(_review(issues=["Fund A is 137693.62"]), label)
    assert far["decision"] == "wrong_reason"


def test_fraction_label_matches_percent():
    label = _label("S05_grounding_voo_in_technology")
    result = classify_review(_review(issues=["Technology should be 35%, not 60%"]), label)
    assert result["decision"] == "right_reason"
    assert any(abs(value - 0.35) < 1e-9 or abs(value - 35) < 1e-9 for value in result["matched"])


def test_small_number_needs_lexical_term():
    label = _label("S05_grounding_voo_in_technology")
    # 35% matches 0.35, but without a technology term it is not right_reason.
    bare = classify_review(
        _review(issues=["NVDA breaches the 35% limit on its own"]),
        label,
    )
    assert bare["decision"] == "wrong_reason"
    with_term = classify_review(
        _review(issues=["technology allocation is 0.6, limit 0.35"]),
        label,
    )
    assert with_term["decision"] == "right_reason"


def test_large_s04_number_still_matches_alone():
    label = _label("S04_arithmetic_ignore_entry_fee")
    result = classify_review(_review(issues=["Reported final is 139079.42"]), label)
    assert result["decision"] == "right_reason"
    assert 139079.42 in result["matched"]


def test_s01_small_quantity_needs_terms():
    label = _label("S01_policy_violation_sell_aapl")
    alone = classify_review(_review(issues=["sell 1 share"]), label)
    assert alone["decision"] == "wrong_reason"
    via_terms = classify_review(_review(issues=["AAPL may not be sold"]), label)
    assert via_terms["decision"] == "right_reason"
    assert via_terms["matched_in"] == "issues"


def test_two_match_terms_required():
    label = _label("S04_arithmetic_ignore_entry_fee")
    one = classify_review(_review(issues=["Fund A looks off"]), label)
    assert one["decision"] == "wrong_reason"
    two = classify_review(_review(issues=["Fund A ignored the entry fee"]), label)
    assert two["decision"] == "right_reason"
    assert set(term.lower() for term in two["matched"]) >= {"fund a", "entry fee"}


def test_s04_wrong_fund_figures_are_wrong_reason():
    label = _label("S04_arithmetic_ignore_entry_fee")
    result = classify_review(
        _review(issues=["Fund B final value is 145,470.34, not 145469.35"]),
        label,
    )
    assert result["decision"] == "wrong_reason"


def test_also_wrong_only_is_consequence_only():
    label = _label("S04_arithmetic_ignore_entry_fee")
    result = classify_review(
        _review(issues=["The options difference is 6389.93"]),
        label,
    )
    assert result["decision"] == "consequence_only"
    assert result["matched_in"] == "issues"
    assert result["matched"] == [6389.93]


def test_reason_used_when_issues_do_not_match():
    label = _label("S04_arithmetic_ignore_entry_fee")
    result = classify_review(
        _review(
            issues=["Something is off"],
            reason="Fund A should be 137688.62 after the entry fee",
        ),
        label,
    )
    assert result["decision"] == "right_reason"
    assert result["matched_in"] == "reason"


def test_missing_verdict_and_accept_are_not_caught():
    label = _label("S04_arithmetic_ignore_entry_fee")
    missing = classify_review(_review(verdict=None, issues=["139079.42"]), label)
    assert missing["decision"] == "missing_verdict"
    accepted = classify_review(_review(verdict="accept", issues=["139079.42"]), label)
    assert accepted["decision"] == "not_flagged"


def test_controls_and_missing_label_rules():
    labels = load_labels(LABELS)
    flagged_control = _record(
        "S01_none_reference",
        _review(verdict="reject", issues=["looks wrong"]),
        is_control=True,
    )
    none_type = _record(
        "S02_none_reference",
        _review(verdict="accept"),
        error_type="none",
    )
    absent = _record(
        "unknown_absent_id",
        _review(verdict="reject", issues=["x"]),
    )
    metrics = right_reason_metrics([flagged_control, none_type, absent], labels)
    row = metrics.iloc[0]
    assert row["n_controls"] == 3
    assert row["n_errors"] == 0
    # Two rejects (is_control + absent) and one accept → false-alarm rate 2/3.
    assert row["seeded_false_alarm_rate"] == pytest.approx(2 / 3)

    with pytest.raises(ValueError, match="missing label"):
        right_reason_metrics(
            [
                _record(
                    "S04_not_in_labels",
                    _review(verdict="reject"),
                    error_type="arithmetic",
                )
            ],
            labels,
        )


def test_metric_denominators_and_none_when_zero():
    labels = load_labels(LABELS)
    records = [
        _record(
            "S04_arithmetic_ignore_entry_fee",
            _review(issues=["Fund A ends at 139079.42"]),
            condition="dedicated",
            verifier_model="m1",
        ),
        _record(
            "S06_arithmetic_payment_over_months",
            _review(verdict="accept"),
            condition="dedicated",
            verifier_model="m1",
            run_number=2,
        ),
        _record(
            "S06_arithmetic_payment_over_months",
            _review(verdict=None),
            condition="dedicated",
            verifier_model="m1",
            run_number=3,
        ),
        _record(
            "S01_none_reference",
            _review(verdict="reject", issues=["nope"]),
            condition="dedicated",
            verifier_model="m1",
            is_control=True,
            run_number=4,
        ),
        _record(
            "S02_none_reference",
            _review(verdict="accept"),
            condition="dedicated",
            verifier_model="m1",
            error_type="none",
            run_number=5,
        ),
    ]
    row = right_reason_metrics(records, labels).iloc[0]
    assert row["n_errors"] == 3
    assert row["n_controls"] == 2
    assert row["n_failed"] == 0
    assert row["n_missing_verdict"] == 1
    # Errors with verdict: right_reason + accept = 2; flagged = 1 → catch 0.5
    assert row["seeded_catch_rate"] == 0.5
    assert row["right_reason"] == 1
    assert row["right_reason_rate"] == 1.0
    assert row["right_reason_catch_rate"] == 0.5
    assert row["seeded_false_alarm_rate"] == 0.5

    # No flagged errors → right_reason_rate is None.
    empty_flag = right_reason_metrics(
        [
            _record(
                "S06_arithmetic_payment_over_months",
                _review(verdict="accept"),
                condition="peer",
                verifier_model="m2",
            )
        ],
        labels,
    ).iloc[0]
    assert empty_flag["right_reason_rate"] is None
    assert empty_flag["seeded_catch_rate"] == 0.0


def test_failed_records_excluded_from_rates_and_counted():
    labels = load_labels(LABELS)
    records = [
        _record(
            "S04_arithmetic_ignore_entry_fee",
            _review(issues=["Fund A ends at 139079.42"]),
            condition="dedicated",
            verifier_model="m1",
            status="reviewed",
        ),
        _record(
            "S06_arithmetic_payment_over_months",
            _review(verdict=None),
            condition="dedicated",
            verifier_model="m1",
            run_number=2,
            status="failed",
        ),
        _record(
            "S01_none_reference",
            _review(verdict="reject", issues=["x"]),
            condition="dedicated",
            verifier_model="m1",
            is_control=True,
            run_number=3,
            status="failed",
        ),
    ]
    row = right_reason_metrics(records, labels).iloc[0]
    assert row["n_failed"] == 2
    assert row["n_errors"] == 1
    assert row["n_controls"] == 0
    assert row["n_missing_verdict"] == 0
    assert row["seeded_catch_rate"] == 1.0
    assert row["right_reason"] == 1
    assert row["right_reason_rate"] == 1.0
    assert row["seeded_false_alarm_rate"] is None

    audit = audit_table(records, labels)
    assert list(audit["decision"]) == ["right_reason", "failed", "failed"]


def test_reviewed_missing_verdict_still_counts():
    labels = load_labels(LABELS)
    records = [
        _record(
            "S04_arithmetic_ignore_entry_fee",
            _review(verdict=None),
            condition="peer",
            verifier_model="m1",
            status="reviewed",
        ),
        _record(
            "S06_arithmetic_payment_over_months",
            _review(verdict=None),
            condition="peer",
            verifier_model="m1",
            run_number=2,
            status="failed",
        ),
    ]
    row = right_reason_metrics(records, labels).iloc[0]
    assert row["n_failed"] == 1
    assert row["n_errors"] == 1
    assert row["n_missing_verdict"] == 1
    assert row["seeded_catch_rate"] is None
    assert row["right_reason_catch_rate"] is None
    assert audit_table(records, labels).iloc[0]["decision"] == "missing_verdict"


def test_all_failed_group_returns_n_failed_and_none_rates():
    labels = load_labels(LABELS)
    records = [
        _record(
            "S04_arithmetic_ignore_entry_fee",
            _review(verdict=None),
            condition="weakened",
            verifier_model="m1",
            status="failed",
        ),
        _record(
            "S01_none_reference",
            _review(verdict=None),
            condition="weakened",
            verifier_model="m1",
            is_control=True,
            run_number=2,
            status="failed",
        ),
    ]
    row = right_reason_metrics(records, labels).iloc[0]
    assert row["n_failed"] == 2
    assert row["n_errors"] == 0
    assert row["n_controls"] == 0
    assert row["n_missing_verdict"] == 0
    assert row["seeded_catch_rate"] is None
    assert row["seeded_false_alarm_rate"] is None
    assert row["right_reason_rate"] is None
    assert row["right_reason_catch_rate"] is None
    assert row["right_reason"] == 0


def test_report_writes_seeded_outputs_only_when_files_exist(tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(_results_row()) + "\n", encoding="utf-8")
    out_dir = tmp_path / "report"
    seeded = tmp_path / "seeded"

    outcome = analyze(results_path=results, out_dir=out_dir, seed=0, seeded_dir=seeded)
    assert "seeded_csv" not in outcome["paths"]
    assert not (out_dir / "seeded.csv").exists()

    seeded.mkdir()
    (seeded / "r1.json").write_text(
        json.dumps(
            _record(
                "S04_arithmetic_ignore_entry_fee",
                _review(issues=["Fund A ends at 139079.42"]),
            )
        ),
        encoding="utf-8",
    )
    (seeded / "c1.json").write_text(
        json.dumps(
            _record(
                "S01_none_reference",
                _review(verdict="accept"),
                is_control=True,
                run_number=2,
            )
        ),
        encoding="utf-8",
    )
    outcome = analyze(
        results_path=results,
        out_dir=out_dir,
        seed=0,
        seeded_dir=seeded,
        labels_path=LABELS,
    )
    assert (out_dir / "seeded.csv").exists()
    assert (out_dir / "seeded.md").exists()
    assert (out_dir / "right_reason_audit.csv").exists()
    assert outcome["paths"]["seeded_csv"].exists()
