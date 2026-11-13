"""Correct reference responses for evaluation.

This module reads ground_truth.json. Its output must never reach an agent.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_GROUND_TRUTH = _ROOT / "data" / "evaluation" / "ground_truth.json"
_SCENARIOS = _ROOT / "data" / "scenarios"


def reference_response(scenario_id: str) -> dict:
    """Return the ground-truth answer formatted as a valid response."""
    ground_truth = _ground_truth(scenario_id)
    metrics = ground_truth["metrics"]
    if scenario_id == "S03":
        response = _contract(ground_truth, observations=_initial_evidence(_scenario("S03")))
        response["violations"] = copy.deepcopy(ground_truth["violations"])
        return response
    if scenario_id == "S04":
        response = _contract(
            ground_truth,
            observations=[
                _money("Fund A", "FINAL", metrics["fund_a_final"]),
                _money("Fund B", "FINAL", metrics["fund_b_final"]),
                _money("OPTIONS", "FINAL", metrics["absolute_difference"], measure="DIFFERENCE"),
            ],
        )
        response["metrics"]["selection"] = metrics["better_fund"]
        return response
    if scenario_id == "S06":
        return _contract(
            ground_truth,
            observations=[
                _money("PORTFOLIO", "FINAL", metrics["final_balance"]),
                _money("PORTFOLIO", "FINAL", metrics["target_shortfall"], measure="DIFFERENCE"),
                _money("PORTFOLIO", "FINAL", metrics["additional_monthly_contribution"], measure="PAYMENT"),
                _money(
                    "PORTFOLIO",
                    "FINAL",
                    metrics["additional_monthly_contribution_cents"],
                    measure="PAYMENT",
                    rounding="UPWARD_CENT",
                ),
            ],
        )
    if scenario_id in {"S09", "S10"}:
        return _contract(ground_truth, observations=_initial_evidence(_scenario(scenario_id)))
    return _contract(ground_truth)


def _ground_truth(scenario_id: str) -> dict:
    return copy.deepcopy(json.loads(_GROUND_TRUTH.read_text(encoding="utf-8"))[scenario_id])


def _scenario(scenario_id: str) -> dict:
    return json.loads((_SCENARIOS / f"{scenario_id}.json").read_text(encoding="utf-8"))


def _contract(ground_truth: dict, observations: list | None = None) -> dict:
    return {
        "scenario_id": ground_truth["scenario_id"],
        "answer_type": ground_truth["answer_type"],
        "status": ground_truth["status"],
        "trades": copy.deepcopy(ground_truth["trades"]),
        "metrics": {
            "observations": [] if observations is None else copy.deepcopy(observations),
            "selection": None,
        },
        "violations": [],
        "explanation": "plan",
    }


def _money(subject: str, stage: str, value, measure: str = "VALUE", rounding: str = "NONE") -> dict:
    return {
        "subject": subject,
        "measure": measure,
        "stage": stage,
        "rounding": rounding,
        "unit": "MONEY",
        "value": value,
    }


def _initial_evidence(scenario: dict) -> list[dict]:
    portfolio = scenario["portfolio"]
    observations = [
        _money("PORTFOLIO", "INITIAL", portfolio["total_value"]),
        _money("CASH", "INITIAL", portfolio["cash_balance"]),
    ]
    for holding in portfolio["holdings"]:
        observations.append(_money(holding["symbol"], "INITIAL", holding["market_value"]))
    return observations
