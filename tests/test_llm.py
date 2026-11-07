"""Tests for the model wrapper. No network: the Gemini client is faked."""

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
