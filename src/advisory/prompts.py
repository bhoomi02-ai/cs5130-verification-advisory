"""Prompts shared by the pilot and the pipeline.

The advisor sees the complete scenario file and nothing else. It never sees
the checker, the answer key or any grade. Keep one copy of each prompt here so
the pilot and every condition use exactly the same text.
"""

from __future__ import annotations

import json

# Bump this whenever ADVISOR_SYSTEM changes, so results record which prompt made them.
# advisor-v1 is the prompt the model pilot (#1) measured the advisor with.
PROMPT_VERSION = "advisor-v1"

ADVISOR_SYSTEM = """\
You are a careful financial advisor working on a synthetic client scenario.
You receive the complete scenario as JSON. Answer the client_request exactly as
asked, respecting every restriction, cash flow and approved assumption in it.

Respond with a single JSON object and nothing else: no prose before or after,
no markdown code fences. The object must conform to
output_requirements.response_schema in the scenario, and follow every rule in
output_requirements (trades_rule, observations_rule, task_completion,
violations_rule, explanation_rule). Check your arithmetic before answering.
"""


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
