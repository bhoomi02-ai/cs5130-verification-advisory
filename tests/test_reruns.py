"""Reruns: usage is counted per attempt, failed runs can be retried, mixed settings are refused.

No network: call_model and check are replaced with fakes. The fake model writes
a log record the way llm.call_model does.
"""

import importlib.util
import json
from datetime import datetime, timedelta, timezone

import pytest

from advisory import llm, pipeline
from advisory.drafts import Settings

SCENARIO = {"scenario_id": "S01", "client_request": "Rebalance to 60/30/10."}
SETTINGS = Settings("gemini-3.1-flash-lite")
DRAFT_ID = "draft-gemini-3.1-flash-lite-S01-r1"
YESTERDAY = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()


def record(run_id, when, tokens=(0, 0), latency=1.0, error=None):
    return {"timestamp": when, "run_id": run_id, "tokens_in": tokens[0], "tokens_out": tokens[1],
            "cost_usd": 0.001 if not error else 0.0, "latency_s": latency, "error": error}


@pytest.fixture
def paths(tmp_path, monkeypatch):
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "S01.json").write_text(json.dumps(SCENARIO))
    monkeypatch.setattr(pipeline, "check", lambda task_id, text: {"correct": True, "format_failure": False, "checks": []})
    return {"results_path": tmp_path / "results.jsonl", "draft_dir": tmp_path / "drafts",
            "scenario_dir": scenarios, "log_path": tmp_path / "runs.jsonl"}


def logging_model(log_path, tokens=(100, 20), latency=2.0):
    def call_model(**kwargs):
        with log_path.open("a") as f:
            f.write(json.dumps(record(kwargs["run_id"], datetime.now(timezone.utc).isoformat(), tokens, latency)) + "\n")
        return '{"status": "FEASIBLE"}'
    return call_model


def test_rerun_after_the_daily_quota_is_not_charged_for_the_first_attempt(monkeypatch, paths):
    # Yesterday's attempt stopped on the daily quota after about 75 s of retries.
    paths["log_path"].write_text(json.dumps(record(DRAFT_ID, YESTERDAY, latency=75.0, error="ClientError: 429")) + "\n")
    monkeypatch.setattr(llm, "call_model", logging_model(paths["log_path"]))
    row = pipeline.run_one("none", "S01", 1, SETTINGS, **paths)
    assert (row["n_calls"], row["n_failed_calls"]) == (1, 0)
    assert (row["tokens_in"], row["tokens_out"], row["latency_s"]) == (100, 20, 2.0)


def test_saved_draft_counts_the_call_that_made_it_once(monkeypatch, paths):
    monkeypatch.setattr(llm, "call_model", logging_model(paths["log_path"], tokens=(300, 50)))
    first = pipeline.run_one("none", "S01", 1, SETTINGS, **paths)
    paths["results_path"].unlink()  # results cleared, the run is done again
    again = pipeline.run_one("none", "S01", 1, SETTINGS, **paths)  # draft comes from disk, no new call
    assert first["tokens_in"] == again["tokens_in"] == 300
    assert again["n_calls"] == 1


def test_results_made_with_other_settings_are_refused(paths):
    paths["results_path"].write_text(json.dumps({
        "task_id": "S01", "condition": "none", "run_number": 1, "model": SETTINGS.model,
        "temperature": 0.2, "top_p": 0.9, "thinking_level": None, "prompt_version": "advisor-v0"}) + "\n")
    with pytest.raises(ValueError, match="other settings"):
        pipeline.done_keys(paths["results_path"], settings=SETTINGS)
    assert pipeline.done_keys(paths["results_path"]) == {("S01", "none", 1, SETTINGS.model)}


def test_drop_failed_removes_only_failed_rows(paths):
    rows = [{"task_id": "S01", "condition": "none", "run_number": 1, "model": "m", "status": "failed"},
            {"task_id": "S02", "condition": "none", "run_number": 1, "model": "m", "status": "unchecked"}]
    paths["results_path"].write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert pipeline.failed_keys(paths["results_path"]) == {("S01", "none", 1, "m")}
    keys = {("S01", "none", 1, "m"), ("S02", "none", 1, "m")}
    assert pipeline.drop_failed(keys, paths["results_path"]) == 1
    assert pipeline.done_keys(paths["results_path"]) == {("S02", "none", 1, "m")}


def test_retry_failed_reruns_only_the_failed_runs(monkeypatch, paths):
    spec = importlib.util.spec_from_file_location("run_condition", pipeline.ROOT / "scripts" / "run_condition.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    base = {"condition": "none", "run_number": 1, "model": SETTINGS.model, "temperature": 0.7, "top_p": 0.9,
            "thinking_level": None, "prompt_version": "advisor-v0"}
    rows = [{**base, "task_id": "S01", "status": "failed"}, {**base, "task_id": "S02", "status": "unchecked"}]
    paths["results_path"].write_text("".join(json.dumps(r) + "\n" for r in rows))
    ran = []

    def fake_run_one(condition, task_id, run_number, settings, results_path):
        ran.append(task_id)
        row = {**base, "task_id": task_id, "status": "unchecked", "initial_correct": True, "final_correct": True,
               "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.0}
        with results_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    monkeypatch.setattr(script, "RESULTS_PATH", paths["results_path"])
    monkeypatch.setattr(script, "run_one", fake_run_one)
    assert script.main(["--condition", "none", "--model", SETTINGS.model, "--tasks", "S01", "S02", "--retry-failed"]) == 0
    assert ran == ["S01"]
    assert pipeline.failed_keys(paths["results_path"]) == set()
    assert len(paths["results_path"].read_text().splitlines()) == 2
