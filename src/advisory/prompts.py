"""Prompts shared by the pilot and the pipeline.

The advisor sees the complete scenario file and nothing else. It never sees
the checker, the answer key or any grade. Keep one copy of each prompt here so
the pilot and every condition use exactly the same text.
"""

from __future__ import annotations

import json

# Bump this whenever ADVISOR_SYSTEM changes, so results record which prompt made them.
PROMPT_VERSION = "advisor-v0"

ADVISOR_SYSTEM = (
    "You are a careful financial advisor. Read the client scenario and answer the "
    "client's request. Follow every rule, restriction and assumption in the scenario, "
    "and check your arithmetic. Reply with a single JSON object that follows the "
    "scenario's output_requirements exactly, with no text before or after it."
)


def advisor_messages(scenario: dict) -> list[dict]:
    """Messages for the advisor's first draft: the system prompt plus the full scenario."""
    return [
        {"role": "system", "content": ADVISOR_SYSTEM},
        {"role": "user", "content": json.dumps(scenario, indent=2)},
    ]


# ---------------------------------------------------------------------------
# Verification (self and peer) and revision.
#
# Self and peer use the SAME review instructions and the SAME revision prompt;
# only who does the review differs, so a difference between the two conditions
# comes from self versus peer, not from wording:
#   self: the advisor's own conversation continues: it sees its answer as its
#         own earlier turn and is asked to re-check it.
#   peer: a fresh, independent call with its own role prompt sees the scenario
#         and the answer as another advisor's work.
# Neither verifier sees the checker, the answer key, a grade or any hint about
# how often answers are wrong. The dedicated verifier (with the calculator)
# gets its own prompt in a later step.
# Bump a version whenever its text changes, so results record which prompt
# made them.

VERIFIER_PROMPT_VERSION = "verifier-v1"
REVISION_PROMPT_VERSION = "revise-v1"

VERDICTS = ("accept", "reject", "abstain")

REVIEW_INSTRUCTIONS = """\
Check the answer against the scenario. Check everything that decides whether it is right:
- It answers the client_request and follows output_requirements and response_schema.
- Every restriction, policy limit, cash requirement and approved assumption in the scenario is respected.
- Every number is correct. Recompute trades, cash, values, allocations and totals from the scenario's own data; do not trust the answer's arithmetic.
- The status (for example FEASIBLE or INFEASIBLE) is right.

Then give exactly one verdict:
- "accept": you checked the answer and found no error.
- "reject": you found at least one specific error. List every error you found, precisely enough for the advisor to fix it.
- "abstain": you cannot decide, for example because the scenario is ambiguous or a figure cannot be checked from the scenario. The answer then goes to a human reviewer.

Judge only against the scenario. Do not reject for style or wording, or because you would have chosen a different plan when the answer's plan also meets every requirement.

Reply with a single JSON object and nothing else (no markdown code fences):
{"verdict": "accept" or "reject" or "abstain", "issues": ["one specific error per item; empty unless the verdict is reject"], "reason": "one or two sentences"}
"""

SELF_REVIEW_REQUEST = (
    "Before your answer above goes to the client, re-check it. For this step only, "
    "reply with a verdict object as described below, not with an answer.\n\n"
    + REVIEW_INSTRUCTIONS
)

PEER_SYSTEM = """\
You are a careful financial advisor reviewing another advisor's answer to a synthetic
client scenario before it goes to the client. You receive the complete scenario as
JSON and the other advisor's answer.
"""

REVISION_REQUEST = """\
A reviewer checked your answer above and rejected it.

Issues the reviewer found:
{issues}

Reviewer's reason: {reason}

Check each issue against the scenario. Fix every real error and return the complete
corrected answer as a single JSON object that follows output_requirements, with no
text before or after it and no markdown code fences. If, after checking, you are sure
an issue is not a real error, keep that part of your answer as it was.
"""


def self_review_messages(scenario: dict, answer: str) -> list[dict]:
    """Self-verification: the advisor's conversation continues and it re-checks its own answer."""
    return advisor_messages(scenario) + [
        {"role": "assistant", "content": answer},
        {"role": "user", "content": SELF_REVIEW_REQUEST},
    ]


def peer_review_messages(scenario: dict, answer: str) -> list[dict]:
    """Peer verification: an independent call reviews the answer as another advisor's work."""
    return [
        {"role": "system", "content": PEER_SYSTEM},
        {
            "role": "user",
            "content": (
                "Scenario:\n"
                + json.dumps(scenario, indent=2)
                + "\n\nThe other advisor's answer:\n"
                + answer
                + "\n\n"
                + REVIEW_INSTRUCTIONS
            ),
        },
    ]


def revision_messages(scenario: dict, answer: str, issues: list[str], reason: str) -> list[dict]:
    """After a reject: the advisor's conversation continues with the reviewer's feedback.

    The same prompt is used after a self or a peer reject.
    """
    listed = "\n".join(f"- {issue}" for issue in issues) or "- (none listed)"
    return advisor_messages(scenario) + [
        {"role": "assistant", "content": answer},
        {"role": "user", "content": REVISION_REQUEST.format(issues=listed, reason=reason or "(none given)")},
    ]


# ---------------------------------------------------------------------------
# Dedicated verifier (with the calculator).
#
# A separate agent with its own role prompt and the sandboxed calculator. It
# uses the same REVIEW_INSTRUCTIONS and verdict format as self and peer, so the
# differences from peer are only the independent verifier role and the tool.
# The tool works through plain JSON replies rather than provider function
# calling, so it runs with any model (including Gemma for the cross-family
# ablation) and every round is an ordinary, logged call_model call.

DEDICATED_PROMPT_VERSION = "dedicated-v1"

# Rounds of calculator requests before the verifier must give its verdict, and
# expressions per round. Together they bound the cost of one verification.
MAX_CALCULATOR_ROUNDS = 3
MAX_EXPRESSIONS_PER_ROUND = 15

DEDICATED_SYSTEM = f"""\
You are an independent verifier at a financial advisory firm. You did not write
the answer you are given. Your only job is to decide whether it is correct before
it reaches the client. You receive the complete client scenario as JSON and the
advisor's answer.

You have a calculator. Use it to recompute every number the answer depends on
instead of doing arithmetic in your head. To use it, reply with only this JSON
object:
{{"calculate": ["<expression>", "<expression>"]}}
with at most {MAX_EXPRESSIONS_PER_ROUND} expressions. You will get the results back
and can ask again, up to {MAX_CALCULATOR_ROUNDS} times in total, so put every
expression you need into as few requests as you can. Calculator rules: + - * / // % **
and parentheses; the functions abs, round, min, max, sum, sqrt, log, log10, exp,
ceil, floor; list literals only inside sum/min/max, e.g. sum([1, 2, 3]);
fv(rate, nper, pmt, pv, when=0), pv(rate, nper, pmt, fv=0, when=0) and
pmt(rate, nper, pv, fv=0, when=0) with numpy-financial conventions (rate per
period, cash paid out is negative; when=0 end of period, 1 start). One expression
per item; no variables, so write the numbers out.

When you have checked everything, reply with your verdict instead.
"""

FINAL_VERDICT_REQUEST = (
    "You have used all your calculator rounds. Give your verdict now, as the "
    "verdict JSON object only."
)


def dedicated_review_messages(scenario: dict, answer: str) -> list[dict]:
    """First message to the dedicated verifier: the full scenario, the answer and the review task."""
    return [
        {"role": "system", "content": DEDICATED_SYSTEM},
        {
            "role": "user",
            "content": (
                "Scenario:\n"
                + json.dumps(scenario, indent=2)
                + "\n\nThe advisor's answer:\n"
                + answer
                + "\n\n"
                + REVIEW_INSTRUCTIONS
            ),
        },
    ]


def calculator_results_message(results: list[dict]) -> dict:
    """The user turn that returns calculator results to the verifier."""
    lines = []
    for i, r in enumerate(results, 1):
        if "result" in r:
            lines.append(f"{i}. {r['expression']} = {r['result']!r}")
        else:
            lines.append(f"{i}. {r['expression']} -> error: {r['error']}")
    return {"role": "user", "content": "Calculator results:\n" + "\n".join(lines)}
