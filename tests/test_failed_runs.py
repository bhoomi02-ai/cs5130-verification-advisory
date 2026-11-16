"""Failed API calls are saved as runs; the daily quota and code errors are not.

No network: call_model is replaced with fakes that raise.
"""

import json

import pytest
from google.genai import errors

from advisory import llm, pipeline
from advisory.drafts import Settings

SCENARIO = {"scenario_id": "S01", "client_request": "Rebalance to 60/30/10."}
SETTINGS = Settings("gemini-3.1-flash-lite")
DAILY_QUOTA = {"error": {"code": 429, "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}


@pytest.fixture
def paths(tmp_path):
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "S01.json").write_text(json.dumps(SCENARIO))
    return {
        "results_path": tmp_path / "results.jsonl",
        "draft_dir": tmp_path / "drafts",
        "scenario_dir": scenarios,
        "log_path": tmp_path / "runs.jsonl",
    }


def raising(error):
    def call_model(**kwargs):
        raise error

    return call_model


def test_api_error_is_saved_as_a_failed_run(monkeypatch, paths):
    monkeypatch.setattr(llm, "call_model", raising(errors.APIError(503, {"error": {"message": "overloaded"}})))
    row = pipeline.run_one("none", "S01", 1, SETTINGS, **paths)

    assert row["status"] == "failed"
    assert "503" in row["error"]
    assert row["verdict"] is None
    assert (row["initial_correct"], row["final_correct"]) == (False, False)
    assert row["run_id"] == "none-gemini-3.1-flash-lite-S01-r1"
    saved = [json.loads(line) for line in paths["results_path"].read_text().splitlines()]
    assert saved == [row]
    # A rerun skips it instead of adding a duplicate row.
    assert ("S01", "none", 1, "gemini-3.1-flash-lite") in pipeline.done_keys(paths["results_path"])


def test_daily_quota_is_not_saved(monkeypatch, paths):
    monkeypatch.setattr(llm, "call_model", raising(errors.APIError(429, DAILY_QUOTA)))
    with pytest.raises(errors.APIError):
        pipeline.run_one("none", "S01", 1, SETTINGS, **paths)
    assert not paths["results_path"].exists()


def test_per_minute_limit_after_retries_is_a_failed_run(monkeypatch, paths):
    minute = {"error": {"code": 429, "details": [{"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}]}]}}
    monkeypatch.setattr(llm, "call_model", raising(errors.APIError(429, minute)))
    row = pipeline.run_one("none", "S01", 1, SETTINGS, **paths)
    assert row["status"] == "failed"


def test_code_error_is_not_saved(monkeypatch, paths):
    monkeypatch.setattr(llm, "call_model", raising(TypeError("unexpected keyword argument 'temperature'")))
    with pytest.raises(TypeError):
        pipeline.run_one("none", "S01", 1, SETTINGS, **paths)
    assert not paths["results_path"].exists()
