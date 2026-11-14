"""self and peer: verify, revise once after a reject, verify again.

No network: call_model replies from a script and check() is faked. Replies come
in call order: the first draft, the first verdict, then (after a reject) the
revision and the second verdict.
"""

import json

import pytest

from advisory import llm, metrics, pipeline, prompts
from advisory.drafts import Settings

SCENARIO = {"scenario_id": "S01", "client_request": "Rebalance to 60/30/10."}
SETTINGS = Settings("gemini-3.6-flash")
DRAFT = '{"status": "FEASIBLE", "trades": ["draft"]}'
REVISED = '{"status": "FEASIBLE", "trades": ["revised"]}'
ACCEPT = '{"verdict": "accept", "issues": [], "reason": "All checks pass."}'
REJECT = '{"verdict": "reject", "issues": ["BND quantity is wrong"], "reason": "One trade is off."}'
ABSTAIN = '{"verdict": "abstain", "issues": [], "reason": "Ambiguous request."}'


@pytest.fixture
def paths(tmp_path, monkeypatch):
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "S01.json").write_text(json.dumps(SCENARIO))
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
            calls.append(kwargs)
            return queue.pop(0)

        monkeypatch.setattr(llm, "call_model", call_model)
        return calls

    return load


def run(condition, paths):
    return pipeline.run_one(condition, "S01", 1, SETTINGS, **paths)


def test_self_accept_keeps_the_draft(script, paths):
    calls = script(DRAFT, ACCEPT)
    row = run("self", paths)
    assert (row["verdict"], row["second_verdict"], row["status"]) == ("accept", None, "accepted")
    assert (row["initial_correct"], row["final_correct"]) == (False, False)
    assert len(calls) == 2
    assert calls[1]["agent"] == "verifier"
    assert calls[1]["messages"] == prompts.self_review_messages(SCENARIO, DRAFT)


def test_peer_reject_then_accept_is_revised(script, paths):
    calls = script(DRAFT, REJECT, REVISED, ACCEPT)
    row = run("peer", paths)
    assert (row["verdict"], row["second_verdict"], row["status"]) == ("reject", "accept", "revised")
    assert (row["initial_correct"], row["final_correct"]) == (False, True)  # fixed
    assert calls[1]["messages"] == prompts.peer_review_messages(SCENARIO, DRAFT)
    assert calls[2]["agent"] == "advisor"
    assert calls[2]["messages"] == prompts.revision_messages(SCENARIO, DRAFT, ["BND quantity is wrong"], "One trade is off.")
    assert calls[3]["messages"] == prompts.peer_review_messages(SCENARIO, REVISED)


def test_second_reject_goes_to_human_review_with_the_revision(script, paths):
    script(DRAFT, REJECT, REVISED, REJECT)
    row = run("self", paths)
    assert (row["verdict"], row["second_verdict"], row["status"]) == ("reject", "reject", "human_review")
    assert row["final_correct"] is True  # the final answer is the revision whatever the second verdict


def test_abstain_goes_to_human_review_with_the_draft(script, paths):
    calls = script(DRAFT, ABSTAIN)
    row = run("peer", paths)
    assert (row["verdict"], row["status"]) == ("abstain", "human_review")
    assert len(calls) == 2  # no revision


def test_unreadable_verdict_is_missing_not_guessed(script, paths):
    calls = script(DRAFT, "I think it looks fine!")
    row = run("self", paths)
    assert (row["verdict"], row["status"]) == (None, "human_review")
    assert len(calls) == 2
    saved = json.loads((paths["results_path"].parent / "answers" / f"{row['run_id']}.json").read_text())
    assert saved["reviews"][0]["error"] and saved["final"] == DRAFT


def test_verifier_and_revision_use_the_advisors_model_and_settings(script, paths):
    calls = script(DRAFT, REJECT, REVISED, ACCEPT)
    run("peer", paths)
    used = {(c["model"], c["temperature"], c["top_p"], c["thinking_level"]) for c in calls}
    assert used == {(SETTINGS.model, SETTINGS.temperature, SETTINGS.top_p, SETTINGS.thinking_level)}
    assert {c["run_id"] for c in calls[1:]} == {"peer-gemini-3.6-flash-S01-r1"}


def test_rows_record_versions_and_the_answers_are_saved(script, paths):
    script(DRAFT, REJECT, REVISED, ACCEPT)
    row = run("peer", paths)
    assert (row["verifier_prompt_version"], row["revision_prompt_version"]) == (
        prompts.VERIFIER_PROMPT_VERSION, prompts.REVISION_PROMPT_VERSION)
    saved = json.loads((paths["results_path"].parent / "answers" / f"{row['run_id']}.json").read_text())
    assert (saved["draft"], saved["revision"], saved["final"]) == (DRAFT, REVISED, REVISED)
    assert [r["stage"] for r in saved["reviews"]] == ["first", "second"]
    assert saved["draft_check"]["correct"] is False and saved["final_check"]["correct"] is True


def test_none_rows_have_no_verifier_versions(script, paths):
    script(DRAFT)
    row = run("none", paths)
    assert (row["verifier_prompt_version"], row["revision_prompt_version"], row["second_verdict"]) == (None, None, None)


def test_results_made_with_other_verifier_prompts_are_refused(paths):
    paths["results_path"].write_text(json.dumps({
        "task_id": "S01", "condition": "self", "run_number": 1, "model": SETTINGS.model,
        "temperature": 0.7, "top_p": 0.9, "thinking_level": None, "prompt_version": prompts.PROMPT_VERSION,
        "verifier_prompt_version": "verifier-v0", "revision_prompt_version": prompts.REVISION_PROMPT_VERSION}) + "\n")
    with pytest.raises(ValueError, match="other verifier prompts"):
        pipeline.done_keys(paths["results_path"], settings=SETTINGS)


def test_metrics_read_none_self_and_peer_rows(script, paths):
    script(DRAFT, ACCEPT)  # none makes the draft; self reuses the saved draft and accepts it
    run("none", paths)
    run("self", paths)
    script(REJECT, REVISED, ACCEPT)  # peer reuses the same saved draft too
    run("peer", paths)
    rows = [json.loads(line) for line in paths["results_path"].read_text().splitlines()]
    summary = metrics.summarize(rows)
    assert set(summary.index.get_level_values("condition")) == {"none", "self", "peer"}
