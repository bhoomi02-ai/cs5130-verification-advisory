"""Groq provider (cross-family verifier). The HTTP call is faked; no network or key needed."""

import pytest

from advisory import llm


def _reply(content="ok", reasoning="", prompt=100, completion=50, total=150):
    return {
        "choices": [{"message": {"role": "assistant", "content": content, "reasoning": reasoning}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total},
    }


@pytest.fixture
def groq(monkeypatch):
    """Replace the HTTP post; each outcome is a reply dict to return or an exception to raise."""

    def install(*outcomes):
        outcomes = list(outcomes)
        sent = []

        def fake_post(body, key):
            sent.append(body)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(llm, "_groq_post", fake_post)
        monkeypatch.setattr(llm.time, "sleep", lambda s: None)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        return sent

    return install


MSGS = [{"role": "system", "content": "You verify."}, {"role": "user", "content": "Check this."}]


def test_system_prompt_moves_into_the_first_user_turn():
    out = llm.to_groq(MSGS + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}])
    assert out == [
        {"role": "user", "content": "You verify.\n\nCheck this."},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "b"},
    ]


def test_groq_model_is_priced_and_dispatched(groq):
    sent = groq(_reply("verdict"))
    text, tokens_in, tokens_out, cost = llm._provider_call("openai/gpt-oss-20b", MSGS, temperature=0.7, top_p=0.9)
    assert text == "verdict" and (tokens_in, tokens_out) == (100, 50)
    assert cost == pytest.approx((100 * 0.075 + 50 * 0.30) / 1_000_000)
    body = sent[0]
    assert body["model"] == "openai/gpt-oss-20b" and body["temperature"] == 0.7 and body["top_p"] == 0.9
    assert body["max_completion_tokens"] == llm.GROQ_MAX_COMPLETION_TOKENS
    assert all(m["role"] != "system" for m in body["messages"])
    assert "reasoning_effort" not in body  # model default unless a level is set


def test_thinking_level_maps_to_reasoning_effort(groq):
    sent = groq(_reply())
    llm._provider_call("openai/gpt-oss-20b", MSGS, thinking_level="low")
    assert sent[0]["reasoning_effort"] == "low"
    with pytest.raises(ValueError, match="thinking_level for Groq"):
        llm._provider_call("openai/gpt-oss-20b", MSGS, thinking_level="minimal")


def test_output_is_never_undercounted():
    # completion_tokens may leave out reasoning: fall back to total - input, then to the text length.
    assert llm.groq_usage_tokens({"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 400}, "", "") == (100, 300)
    assert llm.groq_usage_tokens({"prompt_tokens": 100, "completion_tokens": 0, "total_tokens": 100}, "x" * 40, "y" * 400) == (100, 110)


def test_rate_limit_is_retried_after_the_given_wait(groq, monkeypatch):
    waits = []
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    limited = llm.ProviderError(429, "Rate limit reached for tokens per minute (TPM)")
    limited.retry_after = "12"
    sent = groq(limited, _reply("ok"))
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    text, *_ = llm._provider_call("openai/gpt-oss-20b", MSGS)
    assert text == "ok" and len(sent) == 2 and waits == [12.0]


def test_daily_limit_is_not_retried(groq):
    sent = groq(llm.ProviderError(429, "Rate limit reached for tokens per day (TPD)", daily_quota=True), _reply())
    with pytest.raises(llm.ProviderError):
        llm._provider_call("openai/gpt-oss-20b", MSGS)
    assert len(sent) == 1


def test_client_errors_are_not_retried(groq):
    sent = groq(llm.ProviderError(400, "bad request"), _reply())
    with pytest.raises(llm.ProviderError):
        llm._provider_call("openai/gpt-oss-20b", MSGS)
    assert len(sent) == 1


def test_network_errors_are_retried(groq):
    sent = groq(ConnectionError("reset"), _reply("ok"))
    text, *_ = llm._provider_call("openai/gpt-oss-20b", MSGS)
    assert text == "ok" and len(sent) == 2


def test_missing_key_fails_before_any_request(groq, monkeypatch):
    sent = groq(_reply())
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm._provider_call("openai/gpt-oss-20b", MSGS)
    assert sent == []


def test_groq_calls_count_against_the_budget(groq, monkeypatch, tmp_path):
    log = tmp_path / "runs.jsonl"
    log.write_text('{"cost_usd": 19.95}\n')
    monkeypatch.setattr(llm, "RUN_LOG", log)
    monkeypatch.delenv("ADVISORY_BUDGET_USD", raising=False)
    sent = groq(_reply())
    with pytest.raises(llm.BudgetExceeded):
        llm.call_model(task_id="S04", condition="cross", agent="verifier", model="openai/gpt-oss-20b", messages=MSGS)
    assert sent == []


def _tool_use_failed(arguments):
    import json

    error = llm.ProviderError(400, "tool_use_failed")
    error.body = json.dumps({"error": {"message": "Tool choice is none, but model called a tool",
                                       "code": "tool_use_failed",
                                       "failed_generation": json.dumps({"name": "assistant", "arguments": arguments})}})
    return error


def test_reply_wrapped_as_a_tool_call_is_recovered(groq):
    # Seen live: gpt-oss wraps its calculator request as a call to an undeclared tool.
    sent = groq(_tool_use_failed({"calculate": ["1.0682**5"]}))
    text, tokens_in, tokens_out, cost = llm._provider_call("openai/gpt-oss-20b", MSGS)
    assert text == '{"calculate": ["1.0682**5"]}' and len(sent) == 1  # no retry needed
    assert tokens_out >= llm.GROQ_ESTIMATED_REASONING_TOKENS and tokens_in > 0 and cost > 0  # estimated high


def test_other_bad_requests_are_not_recovered(groq):
    groq(llm.ProviderError(400, "bad request"))
    with pytest.raises(llm.ProviderError):
        llm._provider_call("openai/gpt-oss-20b", MSGS)
