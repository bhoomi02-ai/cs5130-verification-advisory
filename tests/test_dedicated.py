"""Dedicated verifier: prompt and calculator loop. The model is scripted; no API calls."""

import json

import pytest

from advisory import prompts
from advisory.dedicated import calculator_request, dedicated_review
from advisory.verdict import Verdict

SCENARIO = {"scenario_id": "S04", "client_request": "Compare two funds.", "output_requirements": {}}
ANSWER = '{"scenario_id": "S04", "metrics": {"fund_b_final": 145470.34}}'
ACCEPT = json.dumps({"verdict": "accept", "issues": [], "reason": "All numbers check out."})
REJECT = json.dumps({"verdict": "reject", "issues": ["Fund B final value is 145469.35, not 145470.34"], "reason": "x"})


def scripted(*replies):
    """A fake model call that returns `replies` in order and records what it was sent."""
    sent = []

    def call(messages):
        sent.append([dict(m) for m in messages])
        return replies[len(sent) - 1]

    call.sent = sent
    return call


def calc(*expressions):
    return json.dumps({"calculate": list(expressions)})


# Prompt.


def test_dedicated_sees_the_full_scenario_the_answer_and_the_shared_instructions():
    msgs = prompts.dedicated_review_messages(SCENARIO, ANSWER)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == prompts.DEDICATED_SYSTEM
    body = msgs[1]["content"]
    assert json.dumps(SCENARIO, indent=2) in body and ANSWER in body
    assert body.endswith(prompts.REVIEW_INSTRUCTIONS)  # same checklist and verdict format as self and peer


def test_dedicated_prompt_leaks_no_evaluation_details():
    text = (prompts.DEDICATED_SYSTEM + prompts.FINAL_VERDICT_REQUEST).lower()
    for word in ("checker", "answer key", "ground truth", "injected", "seeded", "grade"):
        assert word not in text, word


def test_calculator_results_message_shows_results_and_errors():
    turn = prompts.calculator_results_message(
        [{"expression": "1+1", "result": 2}, {"expression": "foo(1)", "error": "Only these functions"}]
    )
    assert turn == {"role": "user", "content": "Calculator results:\n1. 1+1 = 2\n2. foo(1) -> error: Only these functions"}


# Parsing a calculator request.


@pytest.mark.parametrize(
    "reply, expected",
    [
        (calc("1+1", "2*3"), ["1+1", "2*3"]),
        ('```json\n{"calculate": ["1+1"]}\n```', ["1+1"]),
        ('{"calculate": "1+1"}', ["1+1"]),
        (ACCEPT, None),
        ("not json", None),
        ('{"calculate": 5}', None),
    ],
)
def test_calculator_request(reply, expected):
    assert calculator_request(reply) == expected


# The loop.


def test_verdict_without_calculating():
    call = scripted(ACCEPT)
    review = dedicated_review(SCENARIO, ANSWER, call)
    assert review.verdict == Verdict("accept", "All numbers check out.", [])
    assert review.n_calls == 1 and review.calculations == [] and review.error is None


def test_calculator_results_go_back_to_the_verifier():
    call = scripted(calc("100000 * 1.07784 ** 5", "1/0"), REJECT)
    review = dedicated_review(SCENARIO, ANSWER, call)
    assert review.verdict.verdict == "reject" and review.n_calls == 2
    assert review.calculations[0]["result"] == pytest.approx(145469.34517263246)
    assert "error" in review.calculations[1]  # calculator errors are returned, not raised
    second_call = call.sent[1]
    assert second_call[-2] == {"role": "assistant", "content": calc("100000 * 1.07784 ** 5", "1/0")}
    assert second_call[-1]["content"].startswith("Calculator results:\n1. 100000 * 1.07784 ** 5 = 145469.345")


def test_rounds_are_capped_then_the_verdict_is_requested():
    n = prompts.MAX_CALCULATOR_ROUNDS
    call = scripted(*[calc("1+1")] * n, ACCEPT)
    review = dedicated_review(SCENARIO, ANSWER, call)
    assert review.verdict.verdict == "accept" and review.n_calls == n + 1
    assert call.sent[-1][-1]["content"].endswith(prompts.FINAL_VERDICT_REQUEST)
    assert all(prompts.FINAL_VERDICT_REQUEST not in m[-1]["content"] for m in call.sent[1:-1])


def test_no_verdict_after_the_last_round_is_a_missing_verdict():
    n = prompts.MAX_CALCULATOR_ROUNDS
    review = dedicated_review(SCENARIO, ANSWER, scripted(*[calc("1+1")] * (n + 1)))
    assert review.verdict is None and review.n_calls == n + 1
    assert review.error == "no verdict after the last calculator round"


def test_unreadable_reply_is_a_missing_verdict_not_a_guess():
    review = dedicated_review(SCENARIO, ANSWER, scripted("The answer looks fine."))
    assert review.verdict is None and review.error.startswith("unreadable reply")


def test_too_many_expressions_are_reported_not_evaluated():
    k = prompts.MAX_EXPRESSIONS_PER_ROUND
    review = dedicated_review(SCENARIO, ANSWER, scripted(calc(*["1+1"] * (k + 2)), ACCEPT))
    assert sum("result" in c for c in review.calculations) == k
    assert all("not evaluated" in c["error"] for c in review.calculations[k:])


def test_errors_from_the_model_call_propagate():
    def failing(messages):
        raise RuntimeError("API down")

    with pytest.raises(RuntimeError, match="API down"):
        dedicated_review(SCENARIO, ANSWER, failing)


# Weakened verifier (ablation): dedicated without the calculator.

from advisory.dedicated import weakened_review  # noqa: E402


def test_weakened_prompt_is_the_dedicated_role_without_the_calculator():
    assert prompts.DEDICATED_SYSTEM.startswith(prompts.WEAKENED_SYSTEM.rstrip("\n"))
    assert "calculat" not in prompts.WEAKENED_SYSTEM.lower()
    msgs = prompts.weakened_review_messages(SCENARIO, ANSWER)
    assert msgs[0]["content"] == prompts.WEAKENED_SYSTEM
    # Same user turn as the dedicated verifier: only the system prompt (the tool) differs.
    assert msgs[1] == prompts.dedicated_review_messages(SCENARIO, ANSWER)[1]


def test_weakened_prompt_leaks_no_evaluation_details():
    for word in ("checker", "answer key", "ground truth", "injected", "seeded", "grade"):
        assert word not in prompts.WEAKENED_SYSTEM.lower(), word


def test_weakened_review_is_one_call_with_a_verdict():
    call = scripted(REJECT)
    review = weakened_review(SCENARIO, ANSWER, call)
    assert review.verdict.verdict == "reject" and review.n_calls == 1 and review.calculations == []
    assert len(call.sent) == 1


def test_weakened_review_does_not_run_calculator_requests():
    review = weakened_review(SCENARIO, ANSWER, scripted(calc("1+1")))
    assert review.verdict is None and review.calculations == [] and review.error.startswith("unreadable reply")


# Text after a request or a verdict (seen live with Gemma).

GEMMA_REPLY = (
    '```json\n{"calculate": ["99000 * (1.09 * 0.98)**5", "100000 * (1.08 * 0.998)**5"]}\n```\n\n'
    "**Calculator Results:**\n`100000 * (1.08 * 0.998)**5` = 145470.3447101633\n\n"
    "All numbers match the advisor's answer exactly."
)


def test_text_after_a_calculator_request_is_ignored():
    assert calculator_request(GEMMA_REPLY) == ["99000 * (1.09 * 0.98)**5", "100000 * (1.08 * 0.998)**5"]


def test_made_up_results_never_reach_the_history():
    call = scripted(GEMMA_REPLY, REJECT)
    review = dedicated_review(SCENARIO, ANSWER, call)
    assert review.verdict.verdict == "reject"
    history = call.sent[1]
    assert history[-2] == {"role": "assistant", "content": calc("99000 * (1.09 * 0.98)**5", "100000 * (1.08 * 0.998)**5")}
    assert "145470.3447" not in json.dumps(history)  # the invented result is gone
    assert history[-1]["content"].startswith("Calculator results:\n1. 99000 * (1.09 * 0.98)**5 = 137688.62")


def test_a_reply_that_does_not_start_with_json_is_still_unreadable():
    review = dedicated_review(SCENARIO, ANSWER, scripted('Let me check. {"verdict": "accept"}'))
    assert review.verdict is None


def test_text_dropped_after_a_request_is_kept_for_auditing():
    review = dedicated_review(SCENARIO, ANSWER, scripted(GEMMA_REPLY, ACCEPT))
    assert len(review.ignored) == 1 and review.ignored[0].startswith("**Calculator Results:**")


def test_weakened_user_turn_never_mentions_the_calculator():
    # Locks in that REVIEW_INSTRUCTIONS stays tool-free, so the weakened verifier is never told about a tool.
    msgs = prompts.weakened_review_messages(SCENARIO, ANSWER)
    assert "calculat" not in msgs[1]["content"].lower()
