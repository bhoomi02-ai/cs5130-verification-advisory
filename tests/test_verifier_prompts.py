"""Verifier and revision prompts, and verdict parsing. No model calls."""

import json

import pytest

from advisory import prompts
from advisory.verdict import Verdict, parse_verdict

SCENARIO = {"scenario_id": "S01", "client_request": "Rebalance.", "output_requirements": {}}
DRAFT = '{"scenario_id": "S01", "status": "FEASIBLE", "trades": []}'


def test_self_review_continues_the_advisors_own_conversation():
    msgs = prompts.self_review_messages(SCENARIO, DRAFT)
    assert msgs[:2] == prompts.advisor_messages(SCENARIO)  # same system prompt and scenario as the draft
    assert msgs[2] == {"role": "assistant", "content": DRAFT}  # its own earlier answer
    assert msgs[3]["role"] == "user" and prompts.REVIEW_INSTRUCTIONS in msgs[3]["content"]


def test_peer_review_is_an_independent_call_with_the_full_scenario():
    msgs = prompts.peer_review_messages(SCENARIO, DRAFT)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == prompts.PEER_SYSTEM
    body = msgs[1]["content"]
    assert json.dumps(SCENARIO, indent=2) in body and DRAFT in body
    assert body.endswith(prompts.REVIEW_INSTRUCTIONS)


def test_self_and_peer_use_the_same_review_instructions():
    self_text = prompts.self_review_messages(SCENARIO, DRAFT)[-1]["content"]
    peer_text = prompts.peer_review_messages(SCENARIO, DRAFT)[-1]["content"]
    assert prompts.REVIEW_INSTRUCTIONS in self_text and prompts.REVIEW_INSTRUCTIONS in peer_text


def test_no_verifier_prompt_leaks_evaluation_details():
    text = " ".join([prompts.REVIEW_INSTRUCTIONS, prompts.PEER_SYSTEM, prompts.SELF_REVIEW_REQUEST, prompts.REVISION_REQUEST])
    for word in ("checker", "answer key", "ground truth", "injected", "seeded", "grade"):
        assert word not in text.lower(), word


def test_revision_lists_the_issues_and_keeps_the_conversation():
    msgs = prompts.revision_messages(SCENARIO, DRAFT, ["AAPL was sold", "cash is below $12,000 {min}"], "Breaks a restriction.")
    assert msgs[:3] == prompts.advisor_messages(SCENARIO) + [{"role": "assistant", "content": DRAFT}]
    text = msgs[3]["content"]
    assert "- AAPL was sold\n- cash is below $12,000 {min}" in text  # braces in issues are kept as is
    assert "Reviewer's reason: Breaks a restriction." in text


def test_revision_without_issues_still_reads_sensibly():
    text = prompts.revision_messages(SCENARIO, DRAFT, [], "")[-1]["content"]
    assert "(none listed)" in text and "(none given)" in text


@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"verdict": "accept", "issues": [], "reason": "All checks pass."}', Verdict("accept", "All checks pass.", [])),
        ('```json\n{"verdict": "REJECT", "issues": ["AAPL sold"], "reason": "x"}\n```', Verdict("reject", "x", ["AAPL sold"])),
        ('{"verdict": "abstain", "reason": "ambiguous"}', Verdict("abstain", "ambiguous", [])),
        ('{"verdict": "reject", "issues": "one issue", "reason": ""}', Verdict("reject", "", ["one issue"])),
    ],
)
def test_parse_verdict(reply, expected):
    assert parse_verdict(reply) == expected


@pytest.mark.parametrize(
    "reply",
    ["", "looks fine to me", '["accept"]', '{"verdict": "maybe"}', '{"issues": []}', '{"verdict": "reject", "issues": 3}'],
)
def test_unreadable_verdicts_raise(reply):
    with pytest.raises(ValueError):
        parse_verdict(reply)
