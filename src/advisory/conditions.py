"""The verification conditions.

Every condition starts from the same saved first draft (see drafts.py) and
returns a RunResult with the first draft, the final answer and the verdict.
No condition may see the checker, the answer key or their output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from advisory.drafts import Draft, Settings


@dataclass
class RunResult:
    run_id: str
    task_id: str
    condition: str
    run_number: int
    draft: str  # the advisor's first answer
    final: str  # the answer after verification
    verdict: str | None  # "accept" / "reject" / "abstain"; None when nothing checks the draft
    status: str  # "unchecked" / "accepted" / "revised" / "human_review"


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


def run_self(scenario: dict, draft: Draft, *, run_id: str, settings: Settings) -> RunResult:
    """The advisor reviews and revises its own answer. (Step 3 of #3.)"""
    raise NotImplementedError


def run_peer(scenario: dict, draft: Draft, *, run_id: str, settings: Settings) -> RunResult:
    """A second advisor critiques; the first one revises. (Step 3 of #3.)"""
    raise NotImplementedError


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
