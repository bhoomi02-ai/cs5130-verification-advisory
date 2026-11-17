"""dedicated, weakened and cross_family in the pipeline: same verify-revise-verify flow as self and peer.

No network: call_model replies from a script and check() is faked. Replies come
in call order: the first draft, then the verifier's replies (calculator
requests and the verdict), then after a reject the revision and the second
review's replies.
"""

import json

import pytest

from advisory import conditions, llm, metrics, pipeline, prompts
from advisory.drafts import Settings

SCENARIO = {"scenario_id": "S04", "client_request": "Compare two funds."}
SETTINGS = Settings("gemini-3.6-flash")
DRAFT = '{"status": "FEASIBLE", "metrics": {"fund_b_final": 145470.34}}'
REVISED = '{"status": "FEASIBLE", "metrics": {"fund_b_final": 145469.35}}'
ACCEPT = '{"verdict": "accept", "issues": [], "reason": "All numbers check out."}'
REJECT = '{"verdict": "reject", "issues": ["Fund B final value is 145469.35"], "reason": "Arithmetic slip."}'
CALC = '{"calculate": ["100000 * 1.07784 ** 5"]}'


@pytest.fixture
def paths(tmp_path, monkeypatch):
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "S04.json").write_text(json.dumps(SCENARIO))
    # Only the revised answer is correct, so a reject that leads to a good revision shows up as "fixed".
    monkeypatch.setattr(pipeline, "check", lambda task_id, text: {"correct": text == REVISED, "format_failure": False, "checks": []})
    return {"results_path": tmp_path / "results.jsonl", "draft_dir": tmp_path / "drafts",
            "scenario_dir": scenarios, "log_path": tmp_path / "runs.jsonl"}


@pytest.fixture
def script(monkeypatch):
    calls = []

    def load(*replies):
        queue = list(replies)

        def call_model(**kwargs):
            calls.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})  # a copy: the review conversation grows
            reply = queue.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

        monkeypatch.setattr(llm, "call_model", call_model)
        return calls

    return load


def run(condition, paths):
    return pipeline.run_one(condition, "S04", 1, SETTINGS, **paths)


def saved(row, paths):
    return json.loads((paths["results_path"].parent / "answers" / f"{row['run_id']}.json").read_text())


def test_dedicated_uses_the_calculator_and_revises_after_a_reject(script, paths):
    calls = script(DRAFT, CALC, REJECT, REVISED, ACCEPT)
    row = run("dedicated", paths)
    assert (row["verdict"], row["second_verdict"], row["status"]) == ("reject", "accept", "revised")
    assert (row["initial_correct"], row["final_correct"]) == (False, True)  # fixed
    assert row["verifier_calls"] == 3  # calculator round + verdict, then one call on the revision
    assert calls[1]["agent"] == "verifier" and calls[1]["messages"][:2] == prompts.dedicated_review_messages(SCENARIO, DRAFT)
    assert calls[2]["messages"][-1]["content"].startswith("Calculator results:\n1. 100000 * 1.07784 ** 5 = 145469.345")
    assert calls[3]["agent"] == "advisor"
    assert calls[3]["messages"] == prompts.revision_messages(SCENARIO, DRAFT, ["Fund B final value is 145469.35"], "Arithmetic slip.")
    record = saved(row, paths)
    assert record["reviews"][0]["calculations"][0]["result"] == pytest.approx(145469.34517263246)
    assert [r["calls"] for r in record["reviews"]] == [2, 1]


def test_dedicated_rows_record_the_verifier_setup(script, paths):
    script(DRAFT, ACCEPT)
    row = run("dedicated", paths)
    assert (row["status"], row["verifier_calls"]) == ("accepted", 1)
    assert row["verifier_prompt_version"] == prompts.DEDICATED_PROMPT_VERSION
    assert row["revision_prompt_version"] == prompts.REVISION_PROMPT_VERSION
    assert (row["verifier_model"], row["verifier_thinking_level"]) == (SETTINGS.model, SETTINGS.thinking_level)


def test_weakened_is_one_call_without_the_calculator(script, paths):
    calls = script(DRAFT, REJECT, REVISED, ACCEPT)
    row = run("weakened", paths)
    assert (row["status"], row["verifier_calls"]) == ("revised", 2)
    assert row["verifier_prompt_version"] == prompts.WEAKENED_PROMPT_VERSION
    assert calls[1]["messages"][:2] == prompts.weakened_review_messages(SCENARIO, DRAFT)


def test_weakened_does_not_run_a_calculator_request(script, paths):
    script(DRAFT, CALC)
    row = run("weakened", paths)
    assert (row["verdict"], row["status"], row["verifier_calls"]) == (None, "human_review", 1)
    assert saved(row, paths)["reviews"][0]["calculations"] == []


def test_cross_family_verifier_is_the_other_model_and_the_advisor_revises(script, paths):
    calls = script(DRAFT, CALC, REJECT, REVISED, ACCEPT)
    row = run("cross_family", paths)
    assert row["status"] == "revised" and row["model"] == SETTINGS.model  # the run belongs to the advisor's model
    assert (row["verifier_model"], row["verifier_thinking_level"]) == (
        conditions.CROSS_FAMILY_MODEL, conditions.CROSS_FAMILY_THINKING_LEVEL)
    assert row["verifier_prompt_version"] == prompts.DEDICATED_PROMPT_VERSION
    verifier = [c for c in calls if c["agent"] == "verifier"]
    advisor = [c for c in calls if c["agent"] == "advisor"]
    assert {c["model"] for c in verifier} == {"openai/gpt-oss-20b"} and len(verifier) == 3
    assert {(c["temperature"], c["top_p"], c["thinking_level"]) for c in verifier} == {(0.7, 0.9, "low")}
    assert {c["model"] for c in advisor} == {SETTINGS.model}  # the first draft and the revision
    assert {c["run_id"] for c in calls[1:]} == {"cross_family-gemini-3.6-flash-S04-r1"}


def test_groq_daily_limit_stops_without_saving_a_row(script, paths):
    script(DRAFT, llm.ProviderError(429, "Rate limit reached for tokens per day (TPD)", daily_quota=True))
    with pytest.raises(llm.ProviderError):
        run("cross_family", paths)
    assert not paths["results_path"].exists()


def test_other_groq_errors_are_saved_as_failed_runs(script, paths):
    script(DRAFT, llm.ProviderError(400, "bad request"))
    row = run("cross_family", paths)
    assert row["status"] == "failed" and row["error"].startswith("ProviderError: 400")
    assert row["verifier_calls"] is None and row["verifier_model"] == conditions.CROSS_FAMILY_MODEL


def test_results_made_with_another_verifier_model_are_refused(paths):
    paths["results_path"].write_text(json.dumps({
        "task_id": "S04", "condition": "cross_family", "run_number": 1, "model": SETTINGS.model,
        "temperature": 0.7, "top_p": 0.9, "thinking_level": None, "prompt_version": prompts.PROMPT_VERSION,
        "verifier_prompt_version": prompts.DEDICATED_PROMPT_VERSION, "revision_prompt_version": prompts.REVISION_PROMPT_VERSION,
        "verifier_model": "gemma-4-31b-it", "verifier_thinking_level": None}) + "\n")
    with pytest.raises(ValueError, match="verifier model"):
        pipeline.done_keys(paths["results_path"], settings=SETTINGS)


def test_self_rows_saved_before_verifier_model_was_recorded_still_count(paths):
    paths["results_path"].write_text(json.dumps({
        "task_id": "S04", "condition": "self", "run_number": 1, "model": SETTINGS.model,
        "temperature": 0.7, "top_p": 0.9, "thinking_level": None, "prompt_version": prompts.PROMPT_VERSION,
        "verifier_prompt_version": prompts.VERIFIER_PROMPT_VERSION, "revision_prompt_version": prompts.REVISION_PROMPT_VERSION}) + "\n")
    assert pipeline.done_keys(paths["results_path"], settings=SETTINGS) == {("S04", "self", 1, SETTINGS.model)}


def test_every_condition_shares_one_draft_and_metrics_read_them_all(script, paths):
    calls = script(DRAFT, ACCEPT, ACCEPT, ACCEPT, ACCEPT, ACCEPT)  # one draft, then one accept per checked condition
    for condition in conditions.CONDITIONS:
        run(condition, paths)
    assert sum(c["agent"] == "advisor" for c in calls) == 1  # every condition reused the same saved draft
    rows = [json.loads(line) for line in paths["results_path"].read_text().splitlines()]
    assert {r["condition"] for r in rows} == set(conditions.CONDITIONS)
    summary = metrics.summarize(rows)
    assert set(summary.index.get_level_values("condition")) == set(conditions.CONDITIONS)
