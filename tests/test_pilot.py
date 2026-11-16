"""Pilot runner tests. The model is faked; no network or key needed."""

import json
from pathlib import Path

import pytest
from google.genai import errors

from advisory import pilot

INJECTED = sorted((Path(__file__).resolve().parents[1] / "data" / "injected").glob("*.json"))

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
    """Replace call_model; each reply is a string to return or an exception to raise."""

    def install(*replies):
        replies = list(replies)
        calls = []

        def fake_call_model(**kwargs):
            calls.append(kwargs)
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        monkeypatch.setattr(pilot, "call_model", fake_call_model)
        return calls

    return install


# Parsing and schema validation.


def test_parse_answer_plain_and_fenced():
    assert pilot.parse_answer(json.dumps(GOOD)) == GOOD
    assert pilot.parse_answer("```json\n" + json.dumps(GOOD) + "\n```") == GOOD
    with pytest.raises(ValueError):
        pilot.parse_answer("[1, 2]")


def test_validate_answer_accepts_good_and_names_problems():
    assert pilot.validate_answer(SCENARIO, GOOD) == []
    errors = pilot.validate_answer(SCENARIO, {**GOOD, "status": "MAYBE", "extra": 1})
    assert any("status" in e for e in errors)
    assert any("extra" in e for e in errors)


def test_validate_answer_catches_wrong_scenario_and_type():
    errors = pilot.validate_answer(SCENARIO, {**GOOD, "scenario_id": "S02", "answer_type": "NUMERIC_RESULT"})
    assert any("expected 'S01'" in e for e in errors)
    assert any("expected 'TRADE_PLAN'" in e for e in errors)


# One call per run, first answer scored as it is.


def test_run_advisor_sends_full_scenario_settings_and_run_id(fake_model):
    calls = fake_model(json.dumps(GOOD))
    r = pilot.run_advisor(SCENARIO, "m1", run_id="b-m1-S01-r1", settings={"thinking_level": "low"})
    assert r.schema_valid and r.answer == GOOD and r.api_error is None
    call = calls[0]
    assert json.loads(call["messages"][1]["content"]) == SCENARIO
    assert (call["run_id"], call["condition"], call["task_id"]) == ("b-m1-S01-r1", "pilot", "S01")
    assert (call["temperature"], call["top_p"], call["thinking_level"]) == (0.7, 0.9, "low")
    assert r.settings == {"temperature": 0.7, "top_p": 0.9, "thinking_level": "low"}
    assert r.prompt_version == pilot.PROMPT_VERSION


def test_invalid_first_answer_is_not_retried(fake_model):
    calls = fake_model("Sure! Here is the plan...", json.dumps(GOOD))
    r = pilot.run_advisor(SCENARIO, "m1", run_id="r1")
    assert len(calls) == 1
    assert not r.schema_valid and r.answer is None
    assert r.raw_output == "Sure! Here is the plan..."
    assert "not valid JSON" in r.errors[0]


def _api_error(code=503, quota_id=None):
    details = [{"violations": [{"quotaId": quota_id}]}] if quota_id else []
    return errors.APIError(code, {"error": {"code": code, "message": "unavailable", "details": details}})


def test_api_failure_is_recorded_not_lost(fake_model):
    calls = fake_model(_api_error(503))
    r = pilot.run_advisor(SCENARIO, "m1", run_id="r1")
    assert len(calls) == 1
    assert r.api_error.startswith(("ServerError", "APIError"))
    assert not r.schema_valid and r.correct is None


def test_daily_quota_stops_the_batch(fake_model, tmp_path):
    daily = _api_error(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    calls = fake_model(json.dumps(GOOD), daily, json.dumps(GOOD))
    with pytest.raises(errors.APIError):
        pilot.run_pilot(["m1", "m2", "m3"], [SCENARIO], out_dir=tmp_path, batch_id="b1")
    assert len(calls) == 2  # m3 is never called
    assert (tmp_path / "b1" / "m1" / "S01_run1.json").exists()  # finished runs stay saved
    assert not (tmp_path / "b1" / "m2").exists()


def test_errors_that_are_not_api_errors_are_not_failed_runs(fake_model):
    fake_model(ValueError("No pricing for model 'm1'"))
    with pytest.raises(ValueError, match="No pricing"):
        pilot.run_advisor(SCENARIO, "m1", run_id="r1")


# Batches, grading and the summary.


def test_run_pilot_grades_saves_and_summarizes(fake_model, tmp_path):
    fake_model(json.dumps(GOOD), "garbage", _api_error(503))
    verdict = {
        "correct": False,
        "checks": [
            {"check_type": "REPLAY_TRADES", "passed": False, "details": {"reason": "rule_violation"}},
            {"check_type": "STATUS", "passed": True},
        ],
    }
    results = pilot.run_pilot(
        ["m1", "m2", "m3"], [SCENARIO], grade=lambda s, a: verdict, out_dir=tmp_path, batch_id="b1"
    )
    good, invalid, failed = results
    assert (good.correct, good.failures) == (False, ["REPLAY_TRADES:rule_violation"])
    assert (invalid.correct, invalid.failures) == (False, ["SCHEMA:format"])
    assert (failed.correct, failed.failures) == (None, ["API:error"])
    assert good.run_id == "b1-m1-S01-r1"

    saved = json.loads((tmp_path / "b1" / "m1" / "S01_run1.json").read_text())
    assert saved["run_id"] == "b1-m1-S01-r1" and saved["failures"] == ["REPLAY_TRADES:rule_violation"]

    rows = {row["model"]: row for row in pilot.summarize(results)}
    assert rows["m1"]["accuracy"] == 0.0 and rows["m1"]["api_errors"] == 0
    assert rows["m2"]["schema_valid"] == 0.0 and rows["m2"]["accuracy"] == 0.0
    assert rows["m3"]["api_errors"] == 1 and rows["m3"]["accuracy"] is None and rows["m3"]["graded_runs"] == 0


def test_bool_grader_and_no_grader(fake_model, tmp_path):
    fake_model(json.dumps(GOOD), json.dumps(GOOD))
    graded = pilot.run_pilot(["m1"], [SCENARIO], grade=lambda s, a: True, out_dir=tmp_path, batch_id="b1")
    ungraded = pilot.run_pilot(["m1"], [SCENARIO], out_dir=tmp_path, batch_id="b2")
    assert graded[0].correct is True and graded[0].failures == []
    assert ungraded[0].correct is None and pilot.summarize(ungraded)[0]["accuracy"] is None


def test_batches_never_overwrite(fake_model, tmp_path):
    fake_model(json.dumps(GOOD), json.dumps(GOOD), json.dumps(GOOD))
    pilot.run_pilot(["m1"], [SCENARIO], out_dir=tmp_path, batch_id="b1")
    pilot.run_pilot(["m1"], [SCENARIO], out_dir=tmp_path, batch_id="b2")
    assert (tmp_path / "b1" / "m1" / "S01_run1.json").exists()
    assert (tmp_path / "b2" / "m1" / "S01_run1.json").exists()
    with pytest.raises(FileExistsError):
        pilot.run_pilot(["m1"], [SCENARIO], out_dir=tmp_path, batch_id="b1")


def test_max_calls_refuses_before_any_call(fake_model, tmp_path):
    calls = fake_model()
    with pytest.raises(ValueError, match="6 model calls"):
        pilot.run_pilot(["m1", "m2"], [SCENARIO], repeats=3, max_calls=5, out_dir=tmp_path)
    assert calls == [] and not any(tmp_path.iterdir())


# The real checker on the seeded answers from data/injected/ (#19).


@pytest.mark.parametrize("path", INJECTED, ids=[p.stem for p in INJECTED])
def test_checker_grade_on_injected_answers(path):
    case = json.loads(path.read_text())
    scenario = pilot.load_scenarios(ids=[case["scenario_id"]])[0]
    answer = case["response"]
    assert pilot.validate_answer(scenario, answer) == [], "seeded answers are schema-valid"
    result = pilot.checker_grade(scenario, answer)
    labels = pilot.failure_labels(result)
    if case["error_type"] == "none":
        assert result["correct"] and labels == []
    else:
        assert not result["correct"]
        assert f"{case['expected_check']}:{case['expected_reason']}" in labels


def test_injected_set_is_present():
    kinds = {json.loads(p.read_text())["error_type"] for p in INJECTED}
    assert "none" in kinds and len(kinds) > 3


def test_pilot_uses_the_shared_advisor_prompt(fake_model):
    """The pilot must use the exact prompt of the experiment, or its accuracy would not carry over."""
    from advisory.prompts import ADVISOR_SYSTEM, PROMPT_VERSION, advisor_messages

    calls = fake_model(json.dumps(GOOD))
    r = pilot.run_advisor(SCENARIO, "m1", run_id="r1")
    assert calls[0]["messages"] == advisor_messages(SCENARIO)
    assert calls[0]["messages"][0]["content"] == ADVISOR_SYSTEM
    assert r.prompt_version == PROMPT_VERSION == "advisor-v1"


def test_resume_does_only_the_missing_runs(fake_model, tmp_path):
    daily = _api_error(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    fake_model(json.dumps(GOOD), daily)
    with pytest.raises(errors.APIError):
        pilot.run_pilot(["m1"], [SCENARIO], repeats=3, out_dir=tmp_path, batch_id="b1")

    calls = fake_model(json.dumps(GOOD), json.dumps(GOOD))  # next day: only run2 and run3 are called
    results = pilot.run_pilot(["m1"], [SCENARIO], repeats=3, out_dir=tmp_path, batch_id="b1", resume=True)
    assert len(calls) == 2
    assert [r.run_number for r in results] == [1, 2, 3]
    assert results[0].run_id == "b1-m1-S01-r1"  # loaded from disk, not called again


def test_resume_retries_runs_saved_with_an_api_error(fake_model, tmp_path):
    fake_model(_api_error(503), json.dumps(GOOD))
    pilot.run_pilot(["m1"], [SCENARIO], repeats=2, out_dir=tmp_path, batch_id="b1")
    calls = fake_model(json.dumps(GOOD))
    results = pilot.run_pilot(["m1"], [SCENARIO], repeats=2, out_dir=tmp_path, batch_id="b1", resume=True)
    assert len(calls) == 1 and calls[0]["run_id"] == "b1-m1-S01-r1"
    assert all(r.api_error is None for r in results)


def test_resume_refuses_a_batch_made_with_other_settings(fake_model, tmp_path):
    fake_model(json.dumps(GOOD))
    pilot.run_pilot(["m1"], [SCENARIO], out_dir=tmp_path, batch_id="b1", settings={"temperature": 0.2})
    calls = fake_model()
    with pytest.raises(ValueError, match="start a new batch"):
        pilot.run_pilot(["m1"], [SCENARIO], out_dir=tmp_path, batch_id="b1", resume=True)
    assert calls == []


def test_resume_needs_a_batch_id(tmp_path):
    with pytest.raises(ValueError, match="batch_id"):
        pilot.run_pilot(["m1"], [SCENARIO], out_dir=tmp_path, resume=True)


def test_max_calls_counts_only_the_runs_still_to_do(fake_model, tmp_path):
    fake_model(json.dumps(GOOD), json.dumps(GOOD))
    pilot.run_pilot(["m1"], [SCENARIO], repeats=2, out_dir=tmp_path, batch_id="b1")
    calls = fake_model(json.dumps(GOOD))
    pilot.run_pilot(["m1"], [SCENARIO], repeats=3, out_dir=tmp_path, batch_id="b1", resume=True, max_calls=1)
    assert len(calls) == 1
