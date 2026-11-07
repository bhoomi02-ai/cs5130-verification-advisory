"""Pilot runner tests. The model is faked; no network or key needed."""

import json

import pytest

from advisory import pilot

SCENARIO = {
    "scenario_id": "S01",
    "answer_type": "TRADE_PLAN",
    "client_request": "Rebalance.",
    "output_requirements": {
        "response_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["scenario_id", "answer_type", "status", "trades"],
            "properties": {
                "scenario_id": {"enum": ["S01", "S02"]},
                "answer_type": {"enum": ["TRADE_PLAN", "NUMERIC_RESULT"]},
                "status": {"enum": ["FEASIBLE", "INFEASIBLE"]},
                "trades": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["symbol", "quantity_delta"],
                        "properties": {
                            "symbol": {"type": "string"},
                            "quantity_delta": {"type": "number"},
                        },
                    },
                },
            },
        }
    },
}

GOOD = {
    "scenario_id": "S01",
    "answer_type": "TRADE_PLAN",
    "status": "FEASIBLE",
    "trades": [{"symbol": "MSFT", "quantity_delta": -25}, {"symbol": "BND", "quantity_delta": 125}],
}


@pytest.fixture
def fake_model(monkeypatch):
    def install(*replies):
        replies = list(replies)
        calls = []

        def fake_call_model(**kwargs):
            calls.append(kwargs)
            return replies.pop(0)

        monkeypatch.setattr(pilot, "call_model", fake_call_model)
        return calls

    return install


def test_parse_answer_plain_and_fenced():
    assert pilot.parse_answer(json.dumps(GOOD)) == GOOD
    assert pilot.parse_answer("```json\n" + json.dumps(GOOD) + "\n```") == GOOD
    with pytest.raises(ValueError):
        pilot.parse_answer("[1, 2]")


def test_validate_answer_accepts_good_and_names_problems():
    assert pilot.validate_answer(SCENARIO, GOOD) == []
    bad = {**GOOD, "status": "MAYBE", "extra": 1}
    errors = pilot.validate_answer(SCENARIO, bad)
    assert any("status" in e for e in errors)
    assert any("extra" in e for e in errors)


def test_validate_answer_catches_wrong_scenario_and_type():
    errors = pilot.validate_answer(SCENARIO, {**GOOD, "scenario_id": "S02", "answer_type": "NUMERIC_RESULT"})
    assert any("expected 'S01'" in e for e in errors)
    assert any("expected 'TRADE_PLAN'" in e for e in errors)


def test_run_advisor_valid_first_try(fake_model):
    calls = fake_model(json.dumps(GOOD))
    r = pilot.run_advisor(SCENARIO, "gemini-2.5-flash")
    assert r.schema_valid and r.attempts == 1 and r.answer == GOOD
    # The advisor gets the complete scenario JSON, unchanged.
    assert json.loads(calls[0]["messages"][1]["content"]) == SCENARIO
    assert calls[0]["condition"] == "pilot" and calls[0]["task_id"] == "S01"


def test_run_advisor_retries_once_with_the_problem(fake_model):
    calls = fake_model("Sure! Here is the plan...", json.dumps(GOOD))
    r = pilot.run_advisor(SCENARIO, "gemini-2.5-flash")
    assert r.schema_valid and r.attempts == 2
    fix_msg = calls[1]["messages"][-1]["content"]
    assert "not valid JSON" in fix_msg


def test_run_advisor_gives_up_after_max_attempts(fake_model):
    fake_model("nope", "still nope")
    r = pilot.run_advisor(SCENARIO, "gemini-2.5-flash")
    assert not r.schema_valid and r.answer is None and r.attempts == 2


def test_run_pilot_saves_grades_and_summarizes(fake_model, tmp_path):
    fake_model(json.dumps(GOOD), "garbage", "garbage")
    results = pilot.run_pilot(
        ["m1", "m2"], [SCENARIO], grade=lambda s, a: a["status"] == "FEASIBLE", out_dir=tmp_path
    )
    assert [r.correct for r in results] == [True, False]  # invalid answer counts as wrong
    saved = json.loads((tmp_path / "m1" / "S01_run1.json").read_text())
    assert saved["schema_valid"] and saved["correct"]
    rows = {row["model"]: row for row in pilot.summarize(results)}
    assert rows["m1"]["accuracy"] == 1.0 and rows["m2"]["accuracy"] == 0.0
    assert rows["m2"]["schema_valid"] == 0.0


def test_summarize_without_grader_has_no_accuracy(fake_model, tmp_path):
    fake_model(json.dumps(GOOD))
    rows = pilot.summarize(pilot.run_pilot(["m1"], [SCENARIO], out_dir=tmp_path))
    assert rows[0]["accuracy"] is None and rows[0]["valid_first_try"] == 1.0
