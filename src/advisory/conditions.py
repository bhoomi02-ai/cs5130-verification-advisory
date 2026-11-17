"""The verification conditions.

Every condition starts from the same saved first draft (see drafts.py) and
returns a RunResult with the first draft, the final answer and the verdict.
No condition may see the checker, the answer key or their output.

Every checked condition follows the same flow (from the proposal), and they
differ only in who reviews (see prompts.py and dedicated.py):

1. The verifier checks the draft.
2. accept -> final = draft, status "accepted".
   abstain -> final = draft, status "human_review".
   An unreadable verifier reply -> verdict None (metrics counts it in
   n_missing_verdict), final = draft, status "human_review".
3. reject -> the advisor revises once (revision_messages), then the same
   verifier checks the revision once. Final = the revision whatever the
   second verdict is: accept -> "revised"; reject, abstain or unreadable ->
   "human_review".

The verifiers:

- self: the advisor re-checks its own answer in its own conversation.
- peer: an independent call with the advisor's model.
- dedicated: an independent verifier role with the calculator (up to 3 rounds).
- weakened: the dedicated verifier without the calculator (ablation).
- cross_family: the dedicated verifier played by a model from another company
  (CROSS_FAMILY_MODEL on Groq), with the calculator (ablation).

`verdict` is always the first verdict, which is what catch rate and false
alarms are measured against. Every verifier except cross_family uses the
advisor's model and settings; the revision is always written by the advisor.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

from advisory import llm
from advisory.dedicated import dedicated_review, weakened_review
from advisory.drafts import Draft, Settings
from advisory.prompts import (
    DEDICATED_PROMPT_VERSION,
    VERIFIER_PROMPT_VERSION,
    WEAKENED_PROMPT_VERSION,
    peer_review_messages,
    revision_messages,
    self_review_messages,
)
from advisory.verdict import parse_verdict

# The cross-family verifier: OpenAI's open-weight model on Groq (llm.GROQ_MODELS),
# a model from another company than the advisor's. Low reasoning effort keeps a
# review inside Groq's per-minute token limit.
CROSS_FAMILY_MODEL = "openai/gpt-oss-20b"
CROSS_FAMILY_THINKING_LEVEL = "low"

# The verifier prompt each checked condition uses; rows record it, and
# pipeline.done_keys refuses to mix runs made with another one.
VERIFIER_PROMPTS: dict[str, str] = {
    "self": VERIFIER_PROMPT_VERSION,
    "peer": VERIFIER_PROMPT_VERSION,
    "dedicated": DEDICATED_PROMPT_VERSION,
    "weakened": WEAKENED_PROMPT_VERSION,
    "cross_family": DEDICATED_PROMPT_VERSION,
}

# Conditions whose rows record the verifier model and prompt versions.
VERIFIED = tuple(VERIFIER_PROMPTS)


def verifier_settings(condition: str, settings: Settings) -> Settings | None:
    """The model and settings the verifier calls use: the advisor's, except for cross_family."""
    if condition not in VERIFIED:
        return None
    if condition == "cross_family":
        return replace(settings, model=CROSS_FAMILY_MODEL, thinking_level=CROSS_FAMILY_THINKING_LEVEL)
    return settings


@dataclass
class RunResult:
    run_id: str
    task_id: str
    condition: str
    run_number: int
    draft: str  # the advisor's first answer
    final: str  # the answer after verification
    verdict: str | None  # first verdict: "accept" / "reject" / "abstain"; None when nothing checks the draft or the reply is unreadable
    status: str  # "unchecked" / "accepted" / "revised" / "human_review"
    second_verdict: str | None = None  # the verifier's verdict on the revision, after a reject
    revision: str | None = None  # the revised answer, after a reject
    reviews: list[dict] = field(default_factory=list)  # every review: verdict, issues, raw reply, calculator use
    verifier_calls: int = 0  # model calls made by the verifier in this run (calculator rounds count)


def make_run_id(condition: str, task_id: str, run_number: int, model: str) -> str:
    return f"{condition}-{model}-{task_id}-r{run_number}"


def run_none(scenario: dict, draft: Draft, *, run_id: str, settings: Settings) -> RunResult:
    """No verification: the first draft is the final answer."""
    return RunResult(
        run_id=run_id,
        task_id=draft.task_id,
        condition="none",
        run_number=draft.run_number,
        draft=draft.text,
        final=draft.text,
        verdict=None,
        status="unchecked",
    )


def _call(messages: list[dict], *, agent: str, condition: str, task_id: str, run_id: str, settings: Settings) -> str:
    return llm.call_model(
        task_id=task_id,
        condition=condition,
        agent=agent,
        model=settings.model,
        messages=messages,
        temperature=settings.temperature,
        top_p=settings.top_p,
        thinking_level=settings.thinking_level,
        run_id=run_id,
    )


def _one_call_review(review_messages: Callable) -> Callable:
    """self and peer: one verifier call. An unreadable reply gives verdict None instead of a guess."""

    def review(stage: str, scenario: dict, answer: str, call: Callable[[list[dict]], str]) -> dict:
        text = call(review_messages(scenario, answer))
        try:
            parsed = parse_verdict(text)
        except ValueError as e:
            return {"stage": stage, "verdict": None, "issues": [], "reason": "", "error": str(e), "raw": text, "calls": 1}
        return {
            "stage": stage,
            "verdict": parsed.verdict,
            "issues": parsed.issues,
            "reason": parsed.reason,
            "error": None,
            "raw": text,
            "trailing": getattr(parsed, "trailing", ""),
            "calls": 1,
        }

    return review


def _loop_review(run_review: Callable) -> Callable:
    """dedicated, weakened, cross_family: dedicated.py's review, kept whole for the answers file."""

    def review(stage: str, scenario: dict, answer: str, call: Callable[[list[dict]], str]) -> dict:
        result = run_review(scenario, answer, call)
        v = result.verdict
        return {
            "stage": stage,
            "verdict": v.verdict if v else None,
            "issues": v.issues if v else [],
            "reason": v.reason if v else "",
            "error": result.error,
            "raw": result.messages[-1]["content"] if result.messages else "",
            "trailing": getattr(v, "trailing", "") if v else "",
            "calls": result.n_calls,
            "calculations": result.calculations,
            "ignored": getattr(result, "ignored", []),
            "messages": result.messages,
        }

    return review


REVIEWS: dict[str, Callable] = {
    "self": _one_call_review(self_review_messages),
    "peer": _one_call_review(peer_review_messages),
    "dedicated": _loop_review(dedicated_review),
    "weakened": _loop_review(weakened_review),
    "cross_family": _loop_review(dedicated_review),
}


def _verify_and_revise(scenario: dict, draft: Draft, *, condition: str, run_id: str, settings: Settings) -> RunResult:
    review = REVIEWS[condition]
    ids = {"condition": condition, "task_id": draft.task_id, "run_id": run_id}
    verifier = verifier_settings(condition, settings)

    def verifier_call(messages: list[dict]) -> str:
        return _call(messages, agent="verifier", settings=verifier, **ids)

    first = review("first", scenario, draft.text, verifier_call)
    result = RunResult(
        run_id=run_id,
        task_id=draft.task_id,
        condition=condition,
        run_number=draft.run_number,
        draft=draft.text,
        final=draft.text,
        verdict=first["verdict"],
        status="accepted" if first["verdict"] == "accept" else "human_review",
        reviews=[first],
        verifier_calls=first["calls"],
    )
    if first["verdict"] != "reject":
        return result

    revision = _call(
        revision_messages(scenario, draft.text, first["issues"], first["reason"]), agent="advisor", settings=settings, **ids
    )
    second = review("second", scenario, revision, verifier_call)
    result.revision = revision
    result.final = revision
    result.second_verdict = second["verdict"]
    result.status = "revised" if second["verdict"] == "accept" else "human_review"
    result.reviews.append(second)
    result.verifier_calls += second["calls"]
    return result


def run_self(scenario: dict, draft: Draft, *, run_id: str, settings: Settings) -> RunResult:
    """The advisor re-checks its own answer in its own conversation, and revises once after a reject."""
    return _verify_and_revise(scenario, draft, condition="self", run_id=run_id, settings=settings)


def run_peer(scenario: dict, draft: Draft, *, run_id: str, settings: Settings) -> RunResult:
    """An independent call with the same model reviews the answer; the advisor revises once after a reject."""
    return _verify_and_revise(scenario, draft, condition="peer", run_id=run_id, settings=settings)


def run_dedicated(scenario: dict, draft: Draft, *, run_id: str, settings: Settings) -> RunResult:
    """An independent verifier with the calculator checks the answer; the advisor revises once after a reject."""
    return _verify_and_revise(scenario, draft, condition="dedicated", run_id=run_id, settings=settings)


def run_weakened(scenario: dict, draft: Draft, *, run_id: str, settings: Settings) -> RunResult:
    """Ablation: the dedicated verifier without the calculator (one call per review)."""
    return _verify_and_revise(scenario, draft, condition="weakened", run_id=run_id, settings=settings)


def run_cross_family(scenario: dict, draft: Draft, *, run_id: str, settings: Settings) -> RunResult:
    """Ablation: the dedicated verifier, with the calculator, played by CROSS_FAMILY_MODEL."""
    return _verify_and_revise(scenario, draft, condition="cross_family", run_id=run_id, settings=settings)


ConditionFn = Callable[..., RunResult]

CONDITIONS: dict[str, ConditionFn] = {
    "none": run_none,
    "self": run_self,
    "peer": run_peer,
    "dedicated": run_dedicated,
    "weakened": run_weakened,
    "cross_family": run_cross_family,
}
