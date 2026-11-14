"""The verification conditions.

Every condition starts from the same saved first draft (see drafts.py) and
returns a RunResult with the first draft, the final answer and the verdict.
No condition may see the checker, the answer key or their output.

self and peer follow the same flow (from the proposal), and differ only in
who reviews (see prompts.py):

1. The verifier checks the draft.
2. accept -> final = draft, status "accepted".
   abstain -> final = draft, status "human_review".
   An unreadable verifier reply -> verdict None (metrics counts it in
   n_missing_verdict), final = draft, status "human_review".
3. reject -> the advisor revises once (revision_messages), then the same
   verifier checks the revision once. Final = the revision whatever the
   second verdict is: accept -> "revised"; reject, abstain or unreadable ->
   "human_review".

`verdict` is always the first verdict, which is what catch rate and false
alarms are measured against. The verifier uses the advisor's model and
settings, so self and peer really do use the same model as the advisor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from advisory import llm
from advisory.drafts import Draft, Settings
from advisory.prompts import peer_review_messages, revision_messages, self_review_messages
from advisory.verdict import parse_verdict

# Conditions whose rows record the verifier and revision prompt versions.
VERIFIED = ("self", "peer")


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
    reviews: list[dict] = field(default_factory=list)  # every verifier reply, raw and parsed


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


def _call(messages: list[dict], *, agent: str, condition: str, draft: Draft, run_id: str, settings: Settings) -> str:
    return llm.call_model(
        task_id=draft.task_id,
        condition=condition,
        agent=agent,
        model=settings.model,
        messages=messages,
        temperature=settings.temperature,
        top_p=settings.top_p,
        thinking_level=settings.thinking_level,
        run_id=run_id,
    )


def _review(stage: str, messages: list[dict], **call) -> dict:
    """One verifier call. An unreadable reply gives verdict None instead of a guess."""
    text = _call(messages, agent="verifier", **call)
    try:
        parsed = parse_verdict(text)
    except ValueError as e:
        return {"stage": stage, "verdict": None, "issues": [], "reason": "", "error": str(e), "raw": text}
    return {"stage": stage, "verdict": parsed.verdict, "issues": parsed.issues, "reason": parsed.reason, "error": None, "raw": text}


def _verify_and_revise(
    scenario: dict, draft: Draft, *, condition: str, review_messages: Callable, run_id: str, settings: Settings
) -> RunResult:
    call = {"condition": condition, "draft": draft, "run_id": run_id, "settings": settings}
    first = _review("first", review_messages(scenario, draft.text), **call)
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
    )
    if first["verdict"] != "reject":
        return result

    revision = _call(
        revision_messages(scenario, draft.text, first["issues"], first["reason"]), agent="advisor", **call
    )
    second = _review("second", review_messages(scenario, revision), **call)
    result.revision = revision
    result.final = revision
    result.second_verdict = second["verdict"]
    result.status = "revised" if second["verdict"] == "accept" else "human_review"
    result.reviews.append(second)
    return result


def run_self(scenario: dict, draft: Draft, *, run_id: str, settings: Settings) -> RunResult:
    """The advisor re-checks its own answer in its own conversation, and revises once after a reject."""
    return _verify_and_revise(
        scenario, draft, condition="self", review_messages=self_review_messages, run_id=run_id, settings=settings
    )


def run_peer(scenario: dict, draft: Draft, *, run_id: str, settings: Settings) -> RunResult:
    """An independent call with the same model reviews the answer; the advisor revises once after a reject."""
    return _verify_and_revise(
        scenario, draft, condition="peer", review_messages=peer_review_messages, run_id=run_id, settings=settings
    )


def run_dedicated(scenario: dict, draft: Draft, *, run_id: str, settings: Settings) -> RunResult:
    """A separate verifier with its own prompt and the calculator checks the answer. (Step 4 of #3.)"""
    raise NotImplementedError


ConditionFn = Callable[..., RunResult]

CONDITIONS: dict[str, ConditionFn] = {
    "none": run_none,
    "self": run_self,
    "peer": run_peer,
    "dedicated": run_dedicated,
}
