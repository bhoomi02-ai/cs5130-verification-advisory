"""Deterministic constraint checker. Never shown to any agent."""

from __future__ import annotations

from advisory.schemas import Recommendation


def check(scenario: dict, recommendation: Recommendation) -> dict:
    """Score a recommendation against the scenario's hard constraints."""
    raise NotImplementedError
