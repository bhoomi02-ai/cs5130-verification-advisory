"""Seeded answers with one known error, plus correct controls.

The catalog is fixed. The seed only changes the order in which files are
written. This module reads ground truth and must not be given to an agent.
"""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

from advisory.checker import check
from advisory.evaluation.reference import reference_response

_ROOT = Path(__file__).resolve().parents[2]
_SCENARIOS = _ROOT / "data" / "scenarios"
_GROUND_TRUTH = _ROOT / "data" / "evaluation" / "ground_truth.json"
_DEFAULT_OUTPUT = _ROOT / "data" / "injected"
_SCENARIO_IDS = ("S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10")


class InjectionRejected(RuntimeError):
    """A planned row did not pass its gate, so it was not written."""


def build_catalog(seed: int = 0, output_dir: Path | None = None) -> list[Path]:
    """Write every accepted row and remove catalog files that are no longer produced."""
    output_dir = output_dir or _DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(_catalog())
    random.Random(seed).shuffle(rows)
    records = [_accept(row) for row in rows]
    written = []
    for record in records:
        path = output_dir / record.pop("_filename")
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    keep = {path.name for path in written}
    for path in output_dir.glob("*.json"):
        if path.name not in keep:
            path.unlink()
    return written


def _accept(row: dict) -> dict:
    response = row["response"]
    result = check(row["scenario_id"], response)
    if result["format_failure"]:
        raise InjectionRejected(f"{row['filename']} is a format failure")
    if row["error_type"] == "none":
        if not result["correct"]:
            raise InjectionRejected(f"{row['filename']} was expected to pass")
        return {
            "_filename": row["filename"],
            "scenario_id": row["scenario_id"],
            "error_type": "none",
            "description": row["description"],
            "expected_correct": True,
            "response": response,
        }
    failed = [item for item in result["checks"] if not item["passed"]]
    named = [item for item in failed if item["check_type"] == row["expected_check"]]
    if len(named) != 1 or named[0]["details"].get("reason") != row["expected_reason"]:
        found = [(item["check_type"], item["details"].get("reason")) for item in failed]
        raise InjectionRejected(f"{row['filename']} failed as {found}, not {row['expected_check']}/{row['expected_reason']}")
    also_fails = [
        {"check": item["check_type"], "reason": item["details"].get("reason")}
        for item in failed
        if item["check_type"] != row["expected_check"]
    ]
    return {
        "_filename": row["filename"],
        "scenario_id": row["scenario_id"],
        "error_type": row["error_type"],
        "description": row["description"],
        "expected_check": row["expected_check"],
        "expected_reason": row["expected_reason"],
        "also_fails": also_fails,
        "response": response,
    }


def _catalog() -> list[dict]:
    rows = [_control(scenario_id, "reference", "Untouched reference answer.", reference_response(scenario_id)) for scenario_id in _SCENARIO_IDS]
    rows.extend(
        [
            _control("S01", "alternative", "Alternative valid trade plan.", _with_trades("S01", _s01_alternative_trades())),
            _control("S02", "alternative", "Alternative valid trade plan.", _with_trades("S02", _s02_alternative_trades())),
            _control("S05", "alternative", "Alternative valid trade plan.", _with_trades("S05", _s05_alternative_trades())),
            _control("S07", "alternative", "Alternative valid trade plan.", _with_trades("S07", _s07_alternative_trades())),
        ]
    )
    rows.extend(_error_rows())
    return rows


def _error_rows() -> list[dict]:
    return [
        _error(
            "S01",
            "policy_violation",
            "sell_aapl",
            "Sells 1 share of AAPL, which the policy forbids.",
            "REPLAY_TRADES",
            "rule_violation",
            _with_trades("S01", reference_response("S01")["trades"] + [{"symbol": "AAPL", "quantity_delta": -1}]),
        ),
        _error(
            "S05",
            "policy_violation",
            "technology_36",
            "Leaves direct technology at 36%, one point above the 35% cap.",
            "REPLAY_TRADES",
            "rule_violation",
            _with_trades(
                "S05",
                [
                    {"symbol": "MSFT", "quantity_delta": -60},
                    {"symbol": "VOO", "quantity_delta": 8},
                    {"symbol": "BND", "quantity_delta": 125},
                ],
            ),
        ),
        _error(
            "S02",
            "policy_violation",
            "short_reserve",
            "Sells 20 VOO, which leaves cash below the reserve.",
            "REPLAY_TRADES",
            "rule_violation",
            _with_trades("S02", [{"symbol": "VOO", "quantity_delta": -20}]),
        ),
        _error(
            "S04",
            "arithmetic",
            "ignore_entry_fee",
            "Compounds Fund A from the full investment, without the entry fee.",
            "INDEPENDENT_RESULTS",
            "value_mismatch",
            _s04_without_entry_fee(),
        ),
        _error(
            "S06",
            "arithmetic",
            "payment_over_months",
            "Divides the shortfall by 240 months instead of the annuity factor.",
            "INDEPENDENT_RESULTS",
            "value_mismatch",
            _s06_flat_payment(),
        ),
        _error(
            "S08",
            "arithmetic",
            "skip_withdrawal",
            "Reports final cash as the balance before the withdrawal is paid.",
            "REPLAY_TRADES",
            "value_mismatch",
            _with_observations(
                "S08",
                [_money("CASH", "FINAL", _metrics("S08")["cash_before_withdrawal"])],
            ),
        ),
        _error(
            "S06",
            "grounding",
            "omit_withdrawal",
            "Projects the balance without the month-180 withdrawal.",
            "INDEPENDENT_RESULTS",
            "value_mismatch",
            _s06_without_withdrawal(),
        ),
        _error(
            "S07",
            "grounding",
            "historical_value",
            "Reports the final portfolio value at historical prices.",
            "REPLAY_TRADES",
            "value_mismatch",
            _with_observations(
                "S07",
                [_money("PORTFOLIO", "FINAL", _metrics("S07")["historical_total"])],
            ),
        ),
        _error(
            "S05",
            "grounding",
            "voo_in_technology",
            "Counts VOO in the direct-technology weight.",
            "REPLAY_TRADES",
            "value_mismatch",
            _with_observations("S05", [_fraction("TECHNOLOGY", _s05_tech_plus_voo())]),
        ),
        _error(
            "S01",
            "unit",
            "stock_allocation_percent",
            "Reports the stock allocation as 60 instead of 0.6.",
            "REPLAY_TRADES",
            "value_mismatch",
            _with_observations("S01", [_fraction("STOCK", 60)]),
        ),
        _error(
            "S05",
            "unit",
            "technology_allocation_percent",
            "Reports the technology allocation as 35 instead of 0.35.",
            "REPLAY_TRADES",
            "value_mismatch",
            _with_observations("S05", [_fraction("TECHNOLOGY", 35)]),
        ),
        _error(
            "S08",
            "unit",
            "stock_allocation_percent",
            "Reports the stock allocation as 50 instead of 0.5.",
            "REPLAY_TRADES",
            "value_mismatch",
            _with_observations("S08", [_fraction("STOCK", 50)]),
        ),
        _error(
            "S09",
            "wrong_status",
            "feasible",
            "Reports FEASIBLE for a request that cannot be met.",
            "STATUS",
            "wrong_status",
            _with_status("S09", "FEASIBLE"),
        ),
        _error(
            "S10",
            "wrong_status",
            "feasible",
            "Reports FEASIBLE for a withdrawal that cannot be funded.",
            "STATUS",
            "wrong_status",
            _with_status("S10", "FEASIBLE"),
        ),
        _error(
            "S03",
            "wrong_status",
            "feasible",
            "Reports FEASIBLE while the violation list is still present.",
            "STATUS",
            "wrong_status",
            _with_status("S03", "FEASIBLE"),
        ),
        _error(
            "S03",
            "violation_list",
            "missed_crypto",
            "Leaves out the CRYPTO prohibition.",
            "EXACT_SCOPED_VIOLATIONS",
            "missed_violation",
            _s03_without_crypto(),
        ),
        _error(
            "S03",
            "violation_list",
            "cash_at_limit",
            "Flags cash at exactly 5%, which is the minimum and not a breach.",
            "EXACT_SCOPED_VIOLATIONS",
            "extra_violation",
            _s03_flagging_cash(),
        ),
    ]


def _control(scenario_id: str, slug: str, description: str, response: dict) -> dict:
    return {
        "filename": f"{scenario_id}_none_{slug}.json",
        "scenario_id": scenario_id,
        "error_type": "none",
        "description": description,
        "response": response,
    }


def _error(scenario_id, error_type, slug, description, expected_check, expected_reason, response) -> dict:
    return {
        "filename": f"{scenario_id}_{error_type}_{slug}.json",
        "scenario_id": scenario_id,
        "error_type": error_type,
        "description": description,
        "expected_check": expected_check,
        "expected_reason": expected_reason,
        "response": response,
    }


def _with_trades(scenario_id: str, trades: list[dict]) -> dict:
    response = reference_response(scenario_id)
    response["trades"] = copy.deepcopy(trades)
    return response


def _with_status(scenario_id: str, status: str) -> dict:
    response = reference_response(scenario_id)
    response["status"] = status
    return response


def _with_observations(scenario_id: str, observations: list[dict]) -> dict:
    response = reference_response(scenario_id)
    response["metrics"]["observations"] = observations
    return response


def _s01_alternative_trades():
    # Sell 20 TSLA and 8 VOO. Buy 125 BND. Cash stays 10,000.
    return [
        {"symbol": "TSLA", "quantity_delta": -20},
        {"symbol": "VOO", "quantity_delta": -8},
        {"symbol": "BND", "quantity_delta": 125},
    ]


def _s02_alternative_trades():
    # Sell 238 BND. Final cash is 12,040, above the 12,000 reserve.
    return [{"symbol": "BND", "quantity_delta": -238}]


def _s05_alternative_trades():
    # Sell all NVDA and 40 MSFT. Buy 12 VOO and 125 BND. Technology ends at 34%.
    return [
        {"symbol": "NVDA", "quantity_delta": -50},
        {"symbol": "MSFT", "quantity_delta": -40},
        {"symbol": "VOO", "quantity_delta": 12},
        {"symbol": "BND", "quantity_delta": 125},
    ]


def _s07_alternative_trades():
    # Sell 29 VOO and buy 216 BND. Weights stay inside the 0.5 point band.
    return [
        {"symbol": "VOO", "quantity_delta": -29},
        {"symbol": "BND", "quantity_delta": 216},
    ]


def _s04_without_entry_fee() -> dict:
    scenario = _scenario("S04")
    finals = [_fund_without_entry_fee(option) for option in scenario["investment_options"]]
    response = reference_response("S04")
    response["metrics"]["observations"] = [
        _money("Fund A", "FINAL", finals[0]),
        _money("Fund B", "FINAL", finals[1]),
        _money("OPTIONS", "FINAL", abs(finals[0] - finals[1]), measure="DIFFERENCE"),
    ]
    first, second = scenario["investment_options"]
    response["metrics"]["selection"] = first["name"] if finals[0] > finals[1] else second["name"]
    return response


def _fund_without_entry_fee(option: dict) -> float:
    growth = (1 + option["gross_annual_return"]) * (1 - option["annual_fee"])
    return option["initial_investment"] * growth ** option["years"]


def _s06_flat_payment() -> dict:
    response = reference_response("S06")
    shortfall = next(
        observation["value"]
        for observation in response["metrics"]["observations"]
        if observation["measure"] == "DIFFERENCE"
    )
    periods = _scenario("S06")["retirement_assumptions"]["additional_contribution_duration_months"]
    payment = shortfall / periods
    for observation in response["metrics"]["observations"]:
        if observation["measure"] != "PAYMENT":
            continue
        observation["value"] = _upward_cent(payment) if observation["rounding"] == "UPWARD_CENT" else payment
    return response


def _s06_without_withdrawal() -> dict:
    balance, shortfall, payment, cents = _project_without_withdrawal(_scenario("S06"))
    return _with_observations(
        "S06",
        [
            _money("PORTFOLIO", "FINAL", balance),
            _money("PORTFOLIO", "FINAL", shortfall, measure="DIFFERENCE"),
            _money("PORTFOLIO", "FINAL", payment, measure="PAYMENT"),
            _money("PORTFOLIO", "FINAL", cents, measure="PAYMENT", rounding="UPWARD_CENT"),
        ],
    )


def _project_without_withdrawal(scenario: dict):
    assumptions = scenario["retirement_assumptions"]
    balance = scenario["portfolio"]["total_value"]
    rate = assumptions["monthly_interest_rate"]
    for month in range(1, assumptions["months"] + 1):
        balance *= 1 + rate
        for flow in scenario["cash_flows"]:
            if flow["flow_type"] == "WITHDRAWAL":
                continue
            start = flow["start_month"]
            if start <= month < start + flow["duration_months"]:
                balance += flow["amount"]
    shortfall = assumptions["target"] - balance
    periods = assumptions["additional_contribution_duration_months"]
    payment = shortfall * rate / ((1 + rate) ** periods - 1)
    return balance, shortfall, payment, _upward_cent(payment)


def _s05_tech_plus_voo() -> float:
    metrics = _metrics("S05")
    return metrics["direct_technology_pct"] + metrics["final_values"]["VOO"] / metrics["final_total"]


def _s03_without_crypto() -> dict:
    response = reference_response("S03")
    response["violations"] = [
        row
        for row in response["violations"]
        if not (row["kind"] == "PROHIBITED_CLASS" and row["scope"] == "CRYPTO")
    ]
    return response


def _s03_flagging_cash() -> dict:
    response = reference_response("S03")
    response["violations"] = list(response["violations"]) + [
        {"kind": "CLASS_MIN", "scope": "CASH", "observed": 0.05, "limit": 0.05}
    ]
    return response


def _metrics(scenario_id: str) -> dict:
    return json.loads(_GROUND_TRUTH.read_text(encoding="utf-8"))[scenario_id]["metrics"]


def _scenario(scenario_id: str) -> dict:
    return json.loads((_SCENARIOS / f"{scenario_id}.json").read_text(encoding="utf-8"))


def _money(subject: str, stage: str, value, measure: str = "VALUE", rounding: str = "NONE") -> dict:
    return {
        "subject": subject,
        "measure": measure,
        "stage": stage,
        "rounding": rounding,
        "unit": "MONEY",
        "value": value,
    }


def _fraction(subject: str, value, stage: str = "FINAL") -> dict:
    return {
        "subject": subject,
        "measure": "ALLOCATION",
        "stage": stage,
        "rounding": "NONE",
        "unit": "FRACTION",
        "value": value,
    }


def _upward_cent(amount: float) -> float:
    scaled = amount * 100
    nearest = round(scaled)
    if abs(scaled - nearest) <= 1e-6:
        return nearest / 100
    return math.ceil(scaled - 1e-9) / 100
