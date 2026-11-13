"""Checker tests.

Reference answers come from ground_truth.json. The checker does not read that
file; these tests do, and they compare its recorded results with a fresh
calculation from each scenario.
"""

import json
import math
from pathlib import Path

import pytest

from advisory.checker import check
from advisory.evaluation.reference import reference_response

_ROOT = Path(__file__).resolve().parents[1]
_GROUND_TRUTH = json.loads((_ROOT / "data" / "evaluation" / "ground_truth.json").read_text())
_REPLAY_SCENARIOS = ("S01", "S02", "S05", "S07", "S08")
_RESULT_SCENARIOS = ("S03", "S04", "S06", "S09", "S10")


def _ground_truth(scenario_id):
    return _GROUND_TRUTH[scenario_id]


def _contract(ground_truth, trades=None, status=None, observations=None):
    return {
        "scenario_id": ground_truth["scenario_id"],
        "answer_type": ground_truth["answer_type"],
        "status": ground_truth["status"] if status is None else status,
        "trades": ground_truth["trades"] if trades is None else trades,
        "metrics": {"observations": [] if observations is None else observations, "selection": None},
        "violations": [],
        "explanation": "plan",
    }


def _of_type(result, check_type):
    matches = [row for row in result["checks"] if row["check_type"] == check_type]
    assert len(matches) == 1
    return matches[0]


def _restriction(result, rule, scope):
    matches = [
        row for row in result["restriction_results"] if row["rule"] == rule and row["scope"] == scope
    ]
    assert len(matches) == 1, result["restriction_results"]
    return matches[0]


def _assert_replay_matches_ground_truth(result, ground_truth):
    observed = _of_type(result, "REPLAY_TRADES")["observed"]
    metrics = ground_truth["metrics"]
    for key, expected in metrics.items():
        if key in {"allocations", "final_values", "final_quantities"}:
            continue
        assert key in observed, key
        assert observed[key] == pytest.approx(expected, abs=0.01)
    for name, fraction in metrics["allocations"].items():
        assert observed["allocations"][name] == pytest.approx(fraction, abs=1e-9)
    for symbol, quantity in metrics["final_quantities"].items():
        assert observed["final_quantities"][symbol] == pytest.approx(quantity)
    for symbol, value in metrics["final_values"].items():
        assert observed["final_values"][symbol] == pytest.approx(value, abs=0.01)


@pytest.mark.parametrize("scenario_id", _REPLAY_SCENARIOS)
def test_ground_truth_plan_passes(scenario_id):
    ground_truth = _ground_truth(scenario_id)
    result = check(scenario_id, _contract(ground_truth))
    assert result["format_failure"] is False
    assert result["correct"] is True
    assert {row["check_type"] for row in result["checks"]} == {
        "SCHEMA",
        "REQUIRED_METRICS",
        "STATUS",
        "REPLAY_TRADES",
    }
    _assert_replay_matches_ground_truth(result, ground_truth)


def test_s01_other_valid_plan_passes():
    # Sell 20 TSLA (20 * 300 = 6_000) and 8 VOO (8 * 500 = 4_000).
    # Buy 125 BND (125 * 80 = 10_000). Cash stays 10_000.
    # Stock = 20_000 + 9_000 + 20_000 + 11_000 = 60_000.
    # Bond = 30_000. Each individual stock is at or under 20%. AAPL is untouched.
    trades = [
        {"symbol": "TSLA", "quantity_delta": -20},
        {"symbol": "VOO", "quantity_delta": -8},
        {"symbol": "BND", "quantity_delta": 125},
    ]
    result = check("S01", _contract(_ground_truth("S01"), trades=trades))
    assert result["correct"] is True


def test_s02_other_valid_plan_passes():
    # Sell 238 BND (238 * 80 = 19_040).
    # Cash before the withdrawal = 18_000 + 19_040 = 37_040.
    # Final cash = 37_040 - 25_000 = 12_040, which clears the 12_000 reserve.
    trades = [{"symbol": "BND", "quantity_delta": -238}]
    result = check("S02", _contract(_ground_truth("S02"), trades=trades))
    assert result["correct"] is True
    assert _restriction(result, "MIN_CASH", "CASH")["passed"] is True


def _s05_alternative_trades():
    # Sell all 50 NVDA (10_000) and 40 MSFT (16_000). Sales = 26_000.
    # Buy 12 VOO (6_000) and 125 BND (10_000). Cash ends at 10_000.
    # Direct technology = 20_000 + 14_000 = 34_000, which is 34% of 100_000.
    return [
        {"symbol": "NVDA", "quantity_delta": -50},
        {"symbol": "MSFT", "quantity_delta": -40},
        {"symbol": "VOO", "quantity_delta": 12},
        {"symbol": "BND", "quantity_delta": 125},
    ]


def _technology_observation(value):
    return {
        "subject": "TECHNOLOGY",
        "measure": "ALLOCATION",
        "stage": "FINAL",
        "rounding": "NONE",
        "unit": "FRACTION",
        "value": value,
    }


def test_s05_other_valid_plan_passes():
    result = check("S05", _contract(_ground_truth("S05"), trades=_s05_alternative_trades()))
    assert result["correct"] is True
    tech = _restriction(result, "MAX_DIRECT_SECTOR", "TECHNOLOGY")
    assert tech["passed"] is True
    assert tech["observed"] == pytest.approx(0.34)


def test_s07_other_valid_plan_passes():
    # Deposit brings cash to 14_000 and the book to 112_000.
    # Sell 29 VOO (29 * 500 = 14_500). Buy 216 BND (216 * 80 = 17_280).
    # Cash = 14_000 + 14_500 - 17_280 = 11_220.
    # Stock = 22_000 + 20_000 + 25_500 = 67_500 (60.27%).
    # Bond = 33_280 (29.71%). Cash weight = 10.02%. All inside ±0.5 points.
    trades = [
        {"symbol": "VOO", "quantity_delta": -29},
        {"symbol": "BND", "quantity_delta": 216},
    ]
    result = check("S07", _contract(_ground_truth("S07"), trades=trades))
    assert result["correct"] is True


def test_s08_other_valid_plan_passes():
    # Sell 30 MSFT (12_000) and 36 VOO (18_000). Sales = 30_000.
    # Buy 122 BND (122 * 80 = 9_760).
    # Cash before withdrawal = 10_000 + 30_000 - 9_760 = 30_240.
    # Final cash = 30_240 - 20_000 = 10_240. Book = 80_000.
    # Stock = 40_000 (50%). Bond = 29_760 (37.2%). Cash = 12.8%.
    trades = [
        {"symbol": "MSFT", "quantity_delta": -30},
        {"symbol": "VOO", "quantity_delta": -36},
        {"symbol": "BND", "quantity_delta": 122},
    ]
    result = check("S08", _contract(_ground_truth("S08"), trades=trades))
    assert result["correct"] is True


def test_s01_selling_aapl_fails_no_sell():
    trades = [{"symbol": "AAPL", "quantity_delta": -10}]
    result = check("S01", _contract(_ground_truth("S01"), trades=trades))
    assert result["format_failure"] is False
    assert result["correct"] is False
    row = _restriction(result, "NO_SELL", "AAPL")
    assert row["passed"] is False
    assert row["observed"] == -10
    assert row["limit"] == 0
    assert _of_type(result, "REPLAY_TRADES")["details"]["reason"] == "rule_violation"


def test_s02_short_of_reserve_fails_min_cash():
    # Sell 20 VOO (20 * 500 = 10_000).
    # Cash before withdrawal = 18_000 + 10_000 = 28_000.
    # Final cash = 28_000 - 25_000 = 3_000, still positive, but below 12_000.
    trades = [{"symbol": "VOO", "quantity_delta": -20}]
    result = check("S02", _contract(_ground_truth("S02"), trades=trades))
    assert result["correct"] is False
    row = _restriction(result, "MIN_CASH", "CASH")
    assert row["passed"] is False
    assert row["observed"] == pytest.approx(3000)
    assert row["limit"] == 12000
    assert _restriction(result, "NO_BORROWING", "CASH")["passed"] is True
    assert _of_type(result, "REPLAY_TRADES")["details"]["reason"] == "rule_violation"


def test_s05_technology_left_above_35_percent_fails_sector_cap():
    result = check("S05", _contract(_ground_truth("S05"), trades=[]))
    assert result["correct"] is False
    row = _restriction(result, "MAX_DIRECT_SECTOR", "TECHNOLOGY")
    assert row["passed"] is False
    # AAPL 20_000 + MSFT 30_000 + NVDA 10_000 = 60_000 of the 100_000 book.
    assert row["observed"] == pytest.approx(0.6)
    assert row["limit"] == pytest.approx(0.35)
    assert _of_type(result, "REPLAY_TRADES")["details"]["reason"] == "rule_violation"


def _s05_reference_plus(symbol):
    trades = list(_ground_truth("S05")["trades"]) + [{"symbol": symbol, "quantity_delta": 1}]
    return check("S05", _contract(_ground_truth("S05"), trades=trades))


def _assert_only_unknown_symbol(result, symbol):
    failed = [row for row in result["restriction_results"] if not row["passed"]]
    assert result["format_failure"] is False
    assert result["correct"] is False
    assert [(row["rule"], row["scope"]) for row in failed] == [("EXISTING_ASSETS_ONLY", symbol)]


def test_lowercase_bnd_fails_only_existing_assets():
    # "bnd" is not the holding BND. The share is left out of the book, so the
    # reference plan still meets the technology, stock and class limits.
    _assert_only_unknown_symbol(_s05_reference_plus("bnd"), "bnd")


def test_invented_ticker_fails_only_existing_assets():
    _assert_only_unknown_symbol(_s05_reference_plus("ZZZZ"), "ZZZZ")


def test_s07_selling_aapl_fails_no_sell():
    trades = [{"symbol": "AAPL", "quantity_delta": -1}]
    result = check("S07", _contract(_ground_truth("S07"), trades=trades))
    assert result["correct"] is False
    row = _restriction(result, "NO_SELL", "AAPL")
    assert row["passed"] is False
    assert row["observed"] == -1


def test_s08_selling_aapl_fails_no_sell():
    trades = [{"symbol": "AAPL", "quantity_delta": -1}]
    result = check("S08", _contract(_ground_truth("S08"), trades=trades))
    assert result["correct"] is False
    row = _restriction(result, "NO_SELL", "AAPL")
    assert row["passed"] is False
    assert row["observed"] == -1


@pytest.mark.parametrize(
    "scenario_id, response",
    [
        ("S01", "{"),
        ("S02", {"scenario_id": "S02", "notes": "extra"}),
        ("S05", {"scenario_id": "S05", "answer_type": "TRADE_PLAN", "status": "FEASIBLE", "trades": "none", "metrics": {"observations": [], "selection": None}, "violations": [], "explanation": "plan"}),
        ("S07", {"scenario_id": "S07", "answer_type": "TRADE_PLAN", "status": "FEASIBLE", "trades": [], "metrics": {"observations": [], "selection": None}, "violations": [], "explanation": ""}),
        ("S08", "null"),
    ],
)
def test_malformed_response_is_a_format_failure(scenario_id, response):
    result = check(scenario_id, response)
    assert result["format_failure"] is True
    assert result["correct"] is False
    assert result["restriction_results"] == []
    assert result["checks"][0]["check_type"] == "SCHEMA"
    assert result["checks"][0]["details"]["reason"] == "format"
    assert len(result["checks"]) == 1


def test_wrong_status_is_not_a_format_failure():
    response = _contract(_ground_truth("S01"), status="INFEASIBLE")
    result = check("S01", response)
    assert result["format_failure"] is False
    assert result["correct"] is False
    status = _of_type(result, "STATUS")
    assert status["passed"] is False
    assert status["details"]["reason"] == "wrong_status"


def test_duplicate_symbols_are_netted_and_judged_together():
    # Buying 200 BND first would make cash negative if each trade were applied alone.
    # The net book is the reference plan: BND +125, TSLA -30, VOO -2.
    trades = [
        {"symbol": "BND", "quantity_delta": 200},
        {"symbol": "TSLA", "quantity_delta": -30},
        {"symbol": "VOO", "quantity_delta": -2},
        {"symbol": "BND", "quantity_delta": -75},
    ]
    result = check("S01", _contract(_ground_truth("S01"), trades=trades))
    assert result["correct"] is True
    assert _of_type(result, "REPLAY_TRADES")["observed"]["final_cash"] == pytest.approx(10000)


def test_negative_cash_is_its_own_assumption_row():
    # Buy 200 BND (16_000) with only 10_000 of cash and no offsetting sale.
    trades = [{"symbol": "BND", "quantity_delta": 200}]
    result = check("S01", _contract(_ground_truth("S01"), trades=trades))
    assert result["correct"] is False
    row = _restriction(result, "ASSUMPTION_NO_NEGATIVE_CASH", "CASH")
    assert row["passed"] is False
    assert row["observed"] == pytest.approx(-6000)


def test_reported_allocation_uses_broad_tolerance():
    # 0.6004 is 4e-4 away from the replayed 0.60, inside 5e-4 and outside 1e-6.
    ground_truth = _ground_truth("S01")
    close = _contract(
        ground_truth,
        observations=[
            {
                "subject": "STOCK",
                "measure": "ALLOCATION",
                "stage": "FINAL",
                "rounding": "NONE",
                "unit": "FRACTION",
                "value": 0.6004,
            }
        ],
    )
    assert check("S01", close)["correct"] is True
    far = _contract(
        ground_truth,
        observations=[
            {
                "subject": "STOCK",
                "measure": "ALLOCATION",
                "stage": "FINAL",
                "rounding": "NONE",
                "unit": "FRACTION",
                "value": 0.6006,
            }
        ],
    )
    result = check("S01", far)
    assert result["correct"] is False
    assert _of_type(result, "REPLAY_TRADES")["details"]["reason"] == "value_mismatch"


def test_s05_alternative_reporting_reference_tech_share_mismatches():
    # The alternative book is 34% technology. Reporting the reference plan's 0.35
    # disagrees with this replay by 0.01, which is outside the 5e-4 report tolerance.
    response = _contract(
        _ground_truth("S05"),
        trades=_s05_alternative_trades(),
        observations=[_technology_observation(0.35)],
    )
    result = check("S05", response)
    assert result["correct"] is False
    replay = _of_type(result, "REPLAY_TRADES")
    assert replay["passed"] is False
    assert replay["details"]["reason"] == "value_mismatch"


def test_s05_alternative_reporting_its_own_tech_share_passes():
    response = _contract(
        _ground_truth("S05"),
        trades=_s05_alternative_trades(),
        observations=[_technology_observation(0.34)],
    )
    result = check("S05", response)
    assert result["format_failure"] is False
    assert result["correct"] is True


def _scenario(scenario_id):
    return json.loads((_ROOT / "data" / "scenarios" / f"{scenario_id}.json").read_text())


def _money(subject, stage, value, measure="VALUE", rounding="NONE"):
    return {
        "subject": subject,
        "measure": measure,
        "stage": stage,
        "rounding": rounding,
        "unit": "MONEY",
        "value": value,
    }


def _result_response(scenario_id):
    return reference_response(scenario_id)


def _assert_independent_matches_ground_truth(result, ground_truth):
    expected = _of_type(result, "INDEPENDENT_RESULTS")["observed"]
    for key, recorded in ground_truth["metrics"].items():
        assert key in expected, key
        if key == "allocations":
            for name, fraction in recorded.items():
                assert expected["allocations"][name] == pytest.approx(fraction, abs=1e-9)
        elif key == "additional_monthly_contribution_cents":
            assert expected[key] == recorded
        elif isinstance(recorded, str) or key == "violation_count":
            assert expected[key] == recorded
        elif key.endswith("_pct"):
            assert expected[key] == pytest.approx(recorded, abs=1e-9)
        else:
            assert expected[key] == pytest.approx(recorded, abs=0.01)


@pytest.mark.parametrize("scenario_id", _RESULT_SCENARIOS)
def test_ground_truth_result_passes(scenario_id):
    result = check(scenario_id, _result_response(scenario_id))
    assert result["format_failure"] is False
    assert result["correct"] is True
    _assert_independent_matches_ground_truth(result, _ground_truth(scenario_id))


def test_s03_missing_crypto_violation_fails():
    ground_truth = _ground_truth("S03")
    response = _result_response("S03")
    response["violations"] = [
        row
        for row in ground_truth["violations"]
        if not (row["kind"] == "PROHIBITED_CLASS" and row["scope"] == "CRYPTO")
    ]
    result = check("S03", response)
    assert result["format_failure"] is False
    assert result["correct"] is False
    violations = _of_type(result, "EXACT_SCOPED_VIOLATIONS")
    assert violations["passed"] is False
    assert violations["details"]["reason"] == "missed_violation"


def test_s03_cash_at_five_percent_is_not_a_violation():
    # Cash is 5_000 / 100_000 = 0.05, which is the minimum and not a breach.
    response = _result_response("S03")
    response["violations"] = list(_ground_truth("S03")["violations"]) + [
        {"kind": "CLASS_MIN", "scope": "CASH", "observed": 0.05, "limit": 0.05}
    ]
    result = check("S03", response)
    assert result["format_failure"] is False
    assert result["correct"] is False
    violations = _of_type(result, "EXACT_SCOPED_VIOLATIONS")
    assert violations["passed"] is False
    assert violations["details"]["reason"] == "extra_violation"
    assert {"kind": "CLASS_MIN", "scope": "CASH"} in violations["details"]["extra"]


def _fund_ignoring_entry_fee(option):
    # Same annual factor as the scenario, but the opening value is not reduced by the entry fee.
    growth = (1 + option["gross_annual_return"]) * (1 - option["annual_fee"])
    return option["initial_investment"] * growth ** option["years"]


def test_s04_ignoring_the_entry_fee_fails():
    scenario = _scenario("S04")
    finals = [_fund_ignoring_entry_fee(option) for option in scenario["investment_options"]]
    response = _result_response("S04")
    response["metrics"]["observations"] = [
        _money("Fund A", "FINAL", finals[0]),
        _money("Fund B", "FINAL", finals[1]),
        _money("OPTIONS", "FINAL", abs(finals[0] - finals[1]), measure="DIFFERENCE"),
    ]
    response["metrics"]["selection"] = scenario["investment_options"][0]["name"] if finals[0] > finals[1] else scenario["investment_options"][1]["name"]
    result = check("S04", response)
    assert result["format_failure"] is False
    assert result["correct"] is False
    independent = _of_type(result, "INDEPENDENT_RESULTS")
    assert independent["passed"] is False
    assert independent["details"]["reason"] == "value_mismatch"


def _project_skipping_withdrawals(scenario):
    # Interest, then the month's contributions. The month-180 withdrawal is left out.
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
    cents = math.ceil(payment * 100 - 1e-9) / 100
    return balance, shortfall, payment, cents


def test_s06_omitting_the_month_180_withdrawal_fails():
    balance, shortfall, payment, cents = _project_skipping_withdrawals(_scenario("S06"))
    response = _contract(
        _ground_truth("S06"),
        observations=[
            _money("PORTFOLIO", "FINAL", balance),
            _money("PORTFOLIO", "FINAL", shortfall, measure="DIFFERENCE"),
            _money("PORTFOLIO", "FINAL", payment, measure="PAYMENT"),
            _money("PORTFOLIO", "FINAL", cents, measure="PAYMENT", rounding="UPWARD_CENT"),
        ],
    )
    result = check("S06", response)
    assert result["format_failure"] is False
    assert result["correct"] is False
    independent = _of_type(result, "INDEPENDENT_RESULTS")
    assert independent["passed"] is False
    assert independent["details"]["reason"] == "value_mismatch"


def test_s09_feasible_with_trades_fails():
    response = _result_response("S09")
    response["status"] = "FEASIBLE"
    response["trades"] = [{"symbol": "BND", "quantity_delta": -10}]
    result = check("S09", response)
    assert result["format_failure"] is False
    assert result["correct"] is False
    assert _of_type(result, "STATUS")["details"]["reason"] == "wrong_status"
    assert _of_type(result, "EMPTY_TRADES")["details"]["reason"] == "rule_violation"


def test_s10_proposing_trades_fails():
    response = _result_response("S10")
    response["trades"] = [{"symbol": "VOO", "quantity_delta": -10}]
    result = check("S10", response)
    assert result["format_failure"] is False
    assert result["correct"] is False
    assert _of_type(result, "STATUS")["passed"] is True
    empty = _of_type(result, "EMPTY_TRADES")
    assert empty["passed"] is False
    assert empty["details"]["reason"] == "rule_violation"


def test_s09_missing_symbol_observation_is_not_a_value_mismatch():
    response = _result_response("S09")
    response["metrics"]["observations"] = [
        observation
        for observation in response["metrics"]["observations"]
        if observation["subject"] != "AAPL"
    ]
    result = check("S09", response)
    assert result["format_failure"] is False
    assert result["correct"] is False
    failed = [row for row in result["checks"] if not row["passed"]]
    assert failed
    assert {row["details"]["reason"] for row in failed} == {"missing_observation"}
