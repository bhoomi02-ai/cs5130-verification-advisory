"""Tests for the none condition, saved drafts and the run-and-grade step.

No network: call_model and check are replaced with fakes.
"""

import json

import pytest

from advisory import llm, pipeline
from advisory.conditions import CONDITIONS, run_none
from advisory.drafts import Settings, get_draft
from advisory.prompts import advisor_messages

SCENARIO = {"scenario_id": "S01", "client_request": "Rebalance to 60/30/10."}
GOOD = '{"status": "FEASIBLE", "trades": []}'
SETTINGS = Settings("gemini-3.1-flash-lite")


@pytest.fixture
def fake_model(monkeypatch):
    calls = []

    def call_model(**kwargs):
        calls.append(kwargs)
        return GOOD

    monkeypatch.setattr(llm, "call_model", call_model)
    return calls


@pytest.fixture
def scenario_dir(tmp_path):
    folder = tmp_path / "scenarios"
    folder.mkdir()
    (folder / "S01.json").write_text(json.dumps(SCENARIO))
    return folder


def test_advisor_sees_the_full_scenario_and_nothing_else():
    system, user = advisor_messages(SCENARIO)
    assert system["role"] == "system" and user["role"] == "user"
    assert json.loads(user["content"]) == SCENARIO


def test_draft_is_made_once_then_reused(fake_model, tmp_path):
    first = get_draft(SCENARIO, 1, SETTINGS, root=tmp_path)
    again = get_draft(SCENARIO, 1, SETTINGS, root=tmp_path)
    assert len(fake_model) == 1
    assert first == again and first.text == GOOD
    call = fake_model[0]
    assert (call["condition"], call["agent"], call["run_id"]) == ("draft", "advisor", "draft-gemini-3.1-flash-lite-S01-r1")
    assert (call["temperature"], call["top_p"]) == (0.7, 0.9)


def test_each_run_number_gets_its_own_draft(fake_model, tmp_path):
    get_draft(SCENARIO, 1, SETTINGS, root=tmp_path)
    get_draft(SCENARIO, 2, SETTINGS, root=tmp_path)
    assert len(fake_model) == 2


def test_saved_draft_with_other_settings_is_refused(fake_model, tmp_path):
    get_draft(SCENARIO, 1, SETTINGS, root=tmp_path)
    with pytest.raises(ValueError, match="different draft folder"):
        get_draft(SCENARIO, 1, Settings("gemini-3.1-flash-lite", temperature=0.2), root=tmp_path)


def test_none_keeps_the_draft_as_final(fake_model, tmp_path):
    draft = get_draft(SCENARIO, 1, SETTINGS, root=tmp_path)
    result = run_none(SCENARIO, draft, run_id="none-x-S01-r1", settings=SETTINGS)
    assert result.final == result.draft == GOOD
    assert (result.verdict, result.status, result.condition) == (None, "unchecked", "none")


def test_registry_has_the_four_conditions():
    assert set(CONDITIONS) == {"none", "self", "peer", "dedicated"}


def test_usage_adds_up_draft_and_run_calls(tmp_path):
    log = tmp_path / "runs.jsonl"
    records = [
        {"run_id": "draft-1", "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.001, "latency_s": 1.0, "error": None},
        {"run_id": "none-1", "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.0001, "latency_s": 0.5, "error": None},
        {"run_id": "other", "tokens_in": 999, "tokens_out": 999, "cost_usd": 9.0, "latency_s": 9.0, "error": None},
        {"run_id": "none-1", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "latency_s": 0.2, "error": "APIError"},
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    usage = pipeline.usage_for(["draft-1", "none-1"], log)
    assert (usage["tokens_in"], usage["tokens_out"], usage["n_calls"], usage["n_failed_calls"]) == (110, 55, 3, 1)
    assert usage["cost_usd"] == pytest.approx(0.0011)


def test_run_one_grades_and_saves_a_row(fake_model, tmp_path, scenario_dir, monkeypatch):
    graded = []

    def fake_check(task_id, response):
        graded.append((task_id, response))
        return {"correct": True, "format_failure": False, "checks": [], "restriction_results": []}

    monkeypatch.setattr(pipeline, "check", fake_check)
    results = tmp_path / "results.jsonl"
    row = pipeline.run_one(
        "none", "S01", 1, SETTINGS,
        results_path=results, draft_dir=tmp_path / "drafts", scenario_dir=scenario_dir, log_path=tmp_path / "runs.jsonl",
    )
    assert graded == [("S01", GOOD)]  # none: draft == final, so it is graded once
    assert row["initial_correct"] is True and row["final_correct"] is True
    assert row["run_id"] == "none-gemini-3.1-flash-lite-S01-r1"
    needed = {"task_id", "condition", "run_number", "initial_correct", "final_correct", "verdict",
              "tokens_in", "tokens_out", "cost_usd", "latency_s"}
    assert needed <= set(row)  # the columns metrics.summarize() reads
    assert json.loads(results.read_text()) == row
    assert pipeline.done_keys(results) == {("S01", "none", 1, "gemini-3.1-flash-lite")}


def test_format_failure_is_recorded_as_wrong(fake_model, tmp_path, scenario_dir, monkeypatch):
    monkeypatch.setattr(pipeline, "check", lambda t, r: {"correct": False, "format_failure": True, "checks": []})
    row = pipeline.run_one(
        "none", "S01", 1, SETTINGS,
        results_path=tmp_path / "r.jsonl", draft_dir=tmp_path / "d", scenario_dir=scenario_dir, log_path=tmp_path / "l.jsonl",
    )
    assert row["initial_correct"] is False and row["initial_format_failure"] is True
