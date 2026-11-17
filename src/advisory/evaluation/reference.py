"""Correct reference responses for evaluation.

This module reads ground_truth.json. Its output must never reach an agent.
Responses are built from the scenario's answer type and the evaluator
metric-mapping mode, so new variants (e.g. T03-v001) need no code changes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_GROUND_TRUTH = _ROOT / "data" / "evaluation" / "ground_truth.json"
_METRIC_MAPPING = _ROOT / "data" / "evaluation" / "metric_mapping.json"
_SCENARIOS = _ROOT / "data" / "scenarios"


def reference_response(scenario_id: str) -> dict:
    """Return the ground-truth answer formatted as a valid response."""
    ground_truth = _ground_truth(scenario_id)
    scenario = _scenario(scenario_id)
    metrics = ground_truth["metrics"]
    mode = _mapping_mode(scenario_id)
    answer_type = scenario.get("answer_type") or ground_truth.get("answer_type")

    if answer_type == "VIOLATION_LIST":
        response = _contract(ground_truth, observations=_initial_evidence(scenario))
        response["violations"] = copy.deepcopy(ground_truth["violations"])
        return response

    if mode == "OPTION_RESULTS" or (
        answer_type == "NUMERIC_RESULT" and scenario.get("investment_options")
    ):
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

    if mode == "PROJECTION_RESULTS" or (
        answer_type == "NUMERIC_RESULT" and scenario.get("retirement_assumptions")
    ):
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

    if mode == "PORTFOLIO_EVIDENCE":
        return _contract(ground_truth, observations=_initial_evidence(scenario))

    return _contract(ground_truth)


def _mapping_mode(scenario_id: str) -> str | None:
    mapping = json.loads(_METRIC_MAPPING.read_text(encoding="utf-8"))
    entry = mapping.get("scenarios", {}).get(scenario_id)
    if entry is None:
        return None
    return entry.get("mode")


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
