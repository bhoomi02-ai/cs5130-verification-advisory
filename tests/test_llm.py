"""Tests for the model wrapper. No network: the Gemini client is faked."""

import json
from types import SimpleNamespace

import pytest
from google.genai import errors

from advisory import llm


def _response(text="ok", prompt=100, candidates=20, thoughts=30):
    usage = SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
    )
    return SimpleNamespace(text=text, usage_metadata=usage)


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def fake(monkeypatch):
    def install(*outcomes):
        client = FakeClient(outcomes)
        monkeypatch.setattr(llm, "_get_gemini_client", lambda: client)
        monkeypatch.setattr(llm.time, "sleep", lambda s: None)
        return client

    return install


def test_compute_cost_flash():
    # 1M in at $0.25 + 1M out at $1.50
    assert llm.compute_cost("gemini-3.1-flash-lite", 1_000_000, 1_000_000) == pytest.approx(1.75)


def test_compute_cost_gemma_is_free():
    assert llm.compute_cost("gemma-4-26b-a4b-it", 5000, 5000) == 0.0


def test_compute_cost_unknown_model_raises():
    with pytest.raises(ValueError, match="No pricing"):
        llm.compute_cost("gemini-unknown", 1, 1)


def test_to_gemini_splits_system_and_maps_roles():
    system, contents = llm.to_gemini(
        [
            {"role": "system", "content": "You are an advisor."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
    )
    assert system == "You are an advisor."
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"] == [{"text": "A1"}]


def test_to_gemini_rejects_bad_input():
    with pytest.raises(ValueError):
        llm.to_gemini([{"role": "tool", "content": "x"}])
    with pytest.raises(ValueError):
        llm.to_gemini([{"role": "system", "content": "only system"}])


def test_gemini_call_counts_thinking_as_output(fake):
    client = fake(_response(prompt=1000, candidates=200, thoughts=300))
    text, tin, tout, cost = llm._provider_call(
        "gemini-3.1-flash-lite", [{"role": "user", "content": "hi"}]
    )
    assert (text, tin, tout) == ("ok", 1000, 500)
    assert cost == pytest.approx((1000 * 0.25 + 500 * 1.50) / 1e6)
    assert client.calls[0]["model"] == "gemini-3.1-flash-lite"
    assert client.calls[0]["config"] is None


def test_gemini_call_handles_missing_usage_fields(fake):
    fake(_response(text=None, prompt=None, candidates=None, thoughts=None))
    assert llm._provider_call("gemini-3.1-flash-lite", [{"role": "user", "content": "hi"}]) == (
        "",
        0,
        0,
        0.0,
    )


def test_gemini_call_retries_on_rate_limit(fake):
    client = fake(errors.APIError(429, {}), _response())
    text, *_ = llm._provider_call("gemini-3.1-flash-lite", [{"role": "user", "content": "hi"}])
    assert text == "ok"
    assert len(client.calls) == 2


def test_gemini_call_does_not_retry_client_errors(fake):
    client = fake(errors.APIError(400, {}), _response())
    with pytest.raises(errors.APIError):
        llm._provider_call("gemini-3.1-flash-lite", [{"role": "user", "content": "hi"}])
    assert len(client.calls) == 1


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="No provider"):
        llm._provider_call("gpt-5.4-nano", [{"role": "user", "content": "hi"}])


def test_call_model_logs_record(fake, tmp_path, monkeypatch):
    fake(_response(text="answer"))
    logged = []
    monkeypatch.setattr(llm, "log_record", lambda rec: logged.append(rec))
    out = llm.call_model(
        task_id="S01",
        condition="none",
        agent="advisor",
        model="gemini-3.1-flash-lite",
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
    )
    assert out == "answer"
    rec = logged[0]
    assert (rec.task_id, rec.condition, rec.agent, rec.model) == (
        "S01",
        "none",
        "advisor",
        "gemini-3.1-flash-lite",
    )
    assert rec.tokens_in == 100 and rec.tokens_out == 50
    assert rec.raw_output == "answer"


# Generation settings: temperature, thinking level, daily quota.


def test_gemini_config_defaults_to_none():
    assert llm.gemini_config(None, None, None) is None


def test_gemini_config_is_accepted_by_the_sdk():
    from google.genai import types

    config = llm.gemini_config("sys", 0.7, "low", 0.9)
    assert config == {
        "system_instruction": "sys",
        "temperature": 0.7,
        "top_p": 0.9,
        "thinking_config": {"thinking_level": "LOW"},
    }
    parsed = types.GenerateContentConfig.model_validate(config)
    assert parsed.temperature == 0.7 and parsed.top_p == 0.9
    assert parsed.thinking_config.thinking_level == types.ThinkingLevel.LOW


@pytest.mark.parametrize(
    "temperature, level, top_p",
    [(-0.1, None, None), (2.5, None, None), (None, "max", None), (None, "LOW", None), (None, None, 0), (None, None, 1.5)],
)
def test_gemini_config_rejects_bad_settings(temperature, level, top_p):
    with pytest.raises(ValueError):
        llm.gemini_config(None, temperature, level, top_p)


def test_settings_reach_the_request_and_the_log(fake, monkeypatch):
    client = fake(_response(text="answer"))
    logged = []
    monkeypatch.setattr(llm, "log_record", lambda rec: logged.append(rec))
    llm.call_model(
        task_id="S01",
        condition="none",
        agent="advisor",
        model="gemini-3.1-flash-lite",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        top_p=0.9,
        thinking_level="low",
    )
    assert client.calls[0]["config"] == {
        "temperature": 0.7,
        "top_p": 0.9,
        "thinking_config": {"thinking_level": "LOW"},
    }
    assert (logged[0].temperature, logged[0].top_p, logged[0].thinking_level) == (0.7, 0.9, "low")


def test_daily_quota_is_not_retried(fake):
    daily = errors.APIError(
        429,
        {"error": {"code": 429, "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}},
    )
    client = fake(daily, _response())
    with pytest.raises(errors.APIError):
        llm._provider_call("gemini-3.1-flash-lite", [{"role": "user", "content": "hi"}])
    assert len(client.calls) == 1


def test_per_minute_rate_limit_is_still_retried(fake):
    per_minute = errors.APIError(
        429, {"error": {"code": 429, "details": [{"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}]}]}}
    )
    client = fake(per_minute, _response())
    text, *_ = llm._provider_call("gemini-3.1-flash-lite", [{"role": "user", "content": "hi"}])
    assert text == "ok" and len(client.calls) == 2


def test_run_id_is_logged(fake, monkeypatch):
    fake(_response())
    logged = []
    monkeypatch.setattr(llm, "log_record", lambda rec: logged.append(rec))
    llm.call_model(
        task_id="S01", condition="pilot", agent="advisor", model="gemini-3.1-flash-lite",
        messages=[{"role": "user", "content": "hi"}], run_id="batch1-S01-r1",
    )
    assert logged[0].run_id == "batch1-S01-r1" and logged[0].error is None


def test_failed_call_is_logged_then_raised(fake, monkeypatch):
    fake(errors.APIError(400, {"error": {"message": "bad request"}}))
    logged = []
    monkeypatch.setattr(llm, "log_record", lambda rec: logged.append(rec))
    with pytest.raises(errors.APIError):
        llm.call_model(
            task_id="S01", condition="pilot", agent="advisor", model="gemini-3.1-flash-lite",
            messages=[{"role": "user", "content": "hi"}], run_id="r1", temperature=0.7,
        )
    rec = logged[0]
    assert rec.error.startswith("ClientError") or rec.error.startswith("APIError")
    assert (rec.run_id, rec.temperature, rec.tokens_in, rec.cost_usd, rec.raw_output) == ("r1", 0.7, 0, 0.0, "")


# Spending cap.


def _log_with_spend(path, *costs):
    path.write_text("".join(json.dumps({"cost_usd": c}) + "\n" for c in costs))


def test_budget_defaults_to_20_and_reads_the_env(monkeypatch):
    monkeypatch.delenv("ADVISORY_BUDGET_USD", raising=False)
    assert llm.budget_usd() == 20.0
    monkeypatch.setenv("ADVISORY_BUDGET_USD", "5")
    assert llm.budget_usd() == 5.0


def test_spent_adds_up_the_run_log(tmp_path):
    log = tmp_path / "runs.jsonl"
    assert llm.spent_usd(log) == 0.0
    _log_with_spend(log, 1.5, 0.25, None)
    assert llm.spent_usd(log) == pytest.approx(1.75)


def test_call_is_refused_before_it_could_go_over_budget(fake, monkeypatch, tmp_path):
    log = tmp_path / "runs.jsonl"
    _log_with_spend(log, 19.95)
    monkeypatch.setattr(llm, "RUN_LOG", log)
    monkeypatch.delenv("ADVISORY_BUDGET_USD", raising=False)
    client = fake(_response())
    with pytest.raises(llm.BudgetExceeded, match=r"\$19.95 of the \$20.00"):
        llm.call_model(task_id="S01", condition="none", agent="advisor", model="gemini-3.6-flash",
                       messages=[{"role": "user", "content": "hi"}])
    assert client.calls == []  # nothing sent, nothing spent
    assert llm.spent_usd(log) == pytest.approx(19.95)  # and nothing logged


def test_call_goes_ahead_under_budget(fake, monkeypatch, tmp_path):
    log = tmp_path / "runs.jsonl"
    _log_with_spend(log, 5.0)
    monkeypatch.setattr(llm, "RUN_LOG", log)
    monkeypatch.setattr(llm, "log_record", lambda rec: None)
    monkeypatch.delenv("ADVISORY_BUDGET_USD", raising=False)
    fake(_response())
    assert llm.call_model(task_id="S01", condition="none", agent="advisor", model="gemini-3.6-flash",
                          messages=[{"role": "user", "content": "hi"}]) == "ok"


def test_free_models_are_not_blocked_by_the_budget(fake, monkeypatch, tmp_path):
    log = tmp_path / "runs.jsonl"
    _log_with_spend(log, 20.0)
    monkeypatch.setattr(llm, "RUN_LOG", log)
    monkeypatch.setattr(llm, "log_record", lambda rec: None)
    monkeypatch.delenv("ADVISORY_BUDGET_USD", raising=False)
    fake(_response())
    assert llm.call_model(task_id="S01", condition="cross", agent="verifier", model="gemma-4-31b-it",
                          messages=[{"role": "user", "content": "hi"}]) == "ok"


def test_unpriced_model_fails_before_calling_the_api(fake):
    client = fake(_response())
    with pytest.raises(ValueError, match="No pricing"):
        llm._provider_call("gemini-9-unpriced", [{"role": "user", "content": "hi"}])
    assert client.calls == []  # no request sent, so no tokens spent


# Network errors.


def test_dropped_connection_is_retried(fake):
    import httpx2

    client = fake(httpx2.ReadError("[Errno 54] Connection reset by peer"), _response())
    text, *_ = llm._provider_call("gemini-3.1-flash-lite", [{"role": "user", "content": "hi"}])
    assert text == "ok" and len(client.calls) == 2


def test_network_errors_give_up_after_the_retries(fake):
    client = fake(*[ConnectionResetError("reset")] * llm._MAX_RETRIES)
    with pytest.raises(ConnectionResetError):
        llm._provider_call("gemini-3.1-flash-lite", [{"role": "user", "content": "hi"}])
    assert len(client.calls) == llm._MAX_RETRIES
