"""The four verification conditions.

Each condition takes a scenario and returns a Recommendation. None of them may
see the constraint checker or its output.
"""

from __future__ import annotations

from typing import Callable

from advisory.schemas import Recommendation


def run_none(scenario: dict) -> Recommendation:
    """Advisor answers once; no verification."""
    raise NotImplementedError


def run_self(scenario: dict) -> Recommendation:
    """Advisor reviews and revises its own answer."""
    raise NotImplementedError


def run_peer(scenario: dict) -> Recommendation:
    """A second advisor of the same role critiques; the first revises."""
    raise NotImplementedError


def run_dedicated(scenario: dict) -> Recommendation:
    """A separate verifier agent with its own prompt checks the answer."""
    raise NotImplementedError


CONDITIONS: dict[str, Callable[[dict], Recommendation]] = {
    "none": run_none,
    "self": run_self,
    "peer": run_peer,
    "dedicated": run_dedicated,
}
