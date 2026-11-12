"""Deterministic checker. The only code that reads the evaluation files.

Agents never see ``checks.json`` or ``metric_mapping.json``. Expected numbers
are computed from the scenario. ``ground_truth.json`` is not read here.

``check`` grades one draft. The caller stores ``correct`` as
``initial_correct`` or ``final_correct``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

MONEY_OBSERVATION_TOLERANCE = 0.01

_ROOT = Path(__file__).resolve().parents[2]
_SCENARIOS = _ROOT / "data" / "scenarios"
_EVALUATION = _ROOT / "data" / "evaluation"

_OBSERVATION_FIELDS = ("subject", "measure", "stage", "rounding")


def check(scenario_id: str, response: dict | str) -> dict:
    """Grade one response. A schema failure is incorrect and stops the grade."""
    scenario = _load_json(_SCENARIOS / f"{scenario_id}.json")
    checks = _load_json(_EVALUATION / "checks.json")[scenario_id]
    mapping = _load_json(_EVALUATION / "metric_mapping.json")["scenarios"][scenario_id]
    schema_spec = _spec(checks, "SCHEMA")
    schema_id = schema_spec["check_id"] if schema_spec else f"{scenario_id}_schema"

    parsed, parse_error = _parse_response(response)
    if parse_error:
        return _format_failure(schema_id, [parse_error])
    schema_errors = _schema_errors(parsed, scenario, scenario_id)
    if schema_errors:
        return _format_failure(schema_id, schema_errors)

    replay_spec = _spec(checks, "REPLAY_TRADES")
    if replay_spec:
        ledger = _replay(scenario, parsed["trades"], replay_spec["params_json"])
        computed = ledger["metrics"]
        reported: dict = {}
        analysis = None
    else:
        ledger = None
        analysis = _independent_analysis(scenario, parsed, mapping)
        computed = analysis["expected"]
        reported = analysis["reported"]
    results = []
    restriction_results: list[dict] = []
    for spec in checks:
        kind = spec["check_type"]
        if kind == "SCHEMA":
            results.append(_check_row(spec, True, None, {}))
        elif kind == "REQUIRED_METRICS":
            results.append(_required_metrics(spec, mapping, computed, reported))
        elif kind == "STATUS":
            results.append(_status(spec, parsed))
        elif kind == "REPLAY_TRADES":
            rows, mismatches = _judge_replay(scenario, parsed, ledger, spec["params_json"])
            restriction_results = rows
            results.append(_replay_check(spec, ledger, rows, mismatches))
        elif kind == "INDEPENDENT_RESULTS":
            results.append(_independent_check(spec, analysis))
        elif kind == "EXACT_SCOPED_VIOLATIONS":
            results.append(_exact_violations(spec, scenario, parsed))
        elif kind == "EMPTY_TRADES":
            results.append(_empty_trades(spec, parsed))
        else:
            results.append(
                _check_row(
                    spec,
                    False,
                    None,
                    {"reason": "format", "errors": [f"unsupported check {kind}"]},
                )
            )
    return {
        "correct": all(row["passed"] for row in results),
        "format_failure": False,
        "checks": results,
        "restriction_results": restriction_results,
    }


def _format_failure(check_id: str, errors: list[str]) -> dict:
    return {
        "correct": False,
        "format_failure": True,
        "checks": [
            {
                "check_id": check_id,
                "check_type": "SCHEMA",
                "passed": False,
                "observed": None,
                "details": {"reason": "format", "errors": errors},
            }
        ],
        "restriction_results": [],
    }


def _check_row(spec: dict, passed: bool, observed, details: dict) -> dict:
    return {
        "check_id": spec["check_id"],
        "check_type": spec["check_type"],
        "passed": passed,
        "observed": observed,
        "details": details,
    }


def _required_metrics(spec: dict, mapping: dict, computed: dict, reported: dict) -> dict:
    routes = mapping.get("metric_routes", {})
    missing_observation = []
    not_computed = []
    observed = {}
    for key in spec["params_json"].get("keys", []):
        if _uses_report(routes.get(key, {})):
            value = reported.get(key)
            if value is None:
                missing_observation.append(key)
            else:
                observed[key] = value
            continue
        value = computed.get(key)
        if value is None:
            not_computed.append(key)
        else:
            observed[key] = value
    if missing_observation:
        reason = "missing_observation"
    elif not_computed:
        reason = "rule_violation"
    else:
        reason = None
    details = {}
    if reason:
        details = {
            "reason": reason,
            "missing_observation": missing_observation,
            "not_computed": not_computed,
        }
    return _check_row(spec, reason is None, observed, details)


def _uses_report(route: dict) -> bool:
    """True when the metric is read from the response rather than the scenario."""
    source = str(route.get("source", ""))
    if not source.startswith("REPORTED"):
        return False
    formula = str(route.get("formula", "")).lower()
    return any(token in formula for token in ("reported", "observation", "selection", "violations"))


def _status(spec: dict, response: dict) -> dict:
    expected = spec["params_json"]["expected"]
    actual = response["status"]
    passed = actual == expected
    details = {} if passed else {"reason": "wrong_status", "expected": expected}
    return _check_row(spec, passed, actual, details)


def _replay_check(spec: dict, ledger: dict, rows: list[dict], mismatches: list[dict]) -> dict:
    failed = [row for row in rows if not row["passed"]]
    reasons = []
    items = []
    if failed:
        reasons.append("rule_violation")
        items.extend(
            {
                "reason": "rule_violation",
                "rule": row["rule"],
                "scope": row["scope"],
                "observed": row["observed"],
                "limit": row["limit"],
            }
            for row in failed
        )
    if mismatches:
        reasons.append("value_mismatch")
        items.extend(mismatches)
    details = {}
    if reasons:
        details = {"reason": reasons[0], "reasons": reasons, "items": items}
    return _check_row(spec, not reasons, ledger["observed"], details)


def _judge_replay(scenario: dict, response: dict, ledger: dict, params: dict):
    rows = _restriction_rows(scenario, ledger, params)
    if not any(row["rule"] == "EXISTING_ASSETS_ONLY" for row in rows):
        rows.extend(_unknown_asset_rows(ledger, ledger["roundoff"]))
    rows.extend(ledger["assumption_failures"])
    mismatches = _reported_mismatches(response, ledger, params)
    return rows, mismatches


def _reported_mismatches(response: dict, ledger: dict, params: dict) -> list[dict]:
    """Observations are optional on a trade plan. Present ones must match."""
    money_tolerance = _optional_float(params.get("money_tolerance"), MONEY_OBSERVATION_TOLERANCE)
    fraction_tolerance = _optional_float(params.get("reported_allocation_tolerance"), 0.0)
    expected = ledger["reports"]
    mismatches = []
    for obs in response["metrics"]["observations"]:
        key = tuple(obs[field] for field in _OBSERVATION_FIELDS)
        if key not in expected:
            continue
        unit, value, kind = expected[key]
        tolerance = fraction_tolerance if kind == "FRACTION" else money_tolerance
        reported = obs["value"]
        if obs["unit"] != unit or abs(reported - value) > tolerance:
            mismatches.append(
                {
                    "reason": "value_mismatch",
                    "subject": obs["subject"],
                    "measure": obs["measure"],
                    "stage": obs["stage"],
                    "observed": reported,
                    "expected": value,
                }
            )
    return mismatches


def _replay(scenario: dict, trades: list[dict], params: dict) -> dict:
    """Net each symbol, then value the book once at quoted prices."""
    roundoff = _optional_float(params.get("constraint_roundoff"), 0.0)
    assumption_status = params.get("assumption_status") or "APPROVED"
    approved = {
        item.get("assumption_id")
        for item in scenario.get("assumptions") or []
        if item.get("status") == assumption_status
    }
    closed_ledger = "CLOSED_LONG_ONLY_LEDGER" in approved

    holdings = {item["symbol"]: item for item in scenario["portfolio"]["holdings"]}
    initial_qty = {}
    prices = {}
    for symbol, holding in holdings.items():
        initial_qty[symbol] = _optional_float(holding.get("quantity"), None)
        prices[symbol] = _optional_float(holding.get("current_price"), None)

    net: dict[str, float] = {}
    unknown_net: dict[str, float] = {}
    for trade in trades:
        symbol = trade["symbol"]
        # A name that is not a holding, including a different case, is not replayed.
        bucket = net if symbol in holdings else unknown_net
        bucket[symbol] = bucket.get(symbol, 0.0) + float(trade["quantity_delta"])

    assumption_failures = []
    priced = True
    trade_cost = 0.0
    final_qty = dict(initial_qty)
    for symbol, delta in net.items():
        if abs(delta) <= roundoff:
            continue
        known = symbol in holdings
        price = prices.get(symbol)
        if not known and closed_ledger:
            assumption_failures.append(_rule("ASSUMPTION_NO_NEW_SYMBOL", symbol, False, delta, 0))
        if price is None:
            priced = False
            assumption_failures.append(_rule("ASSUMPTION_QUOTED_PRICE", symbol, False, None, None))
            continue
        start = initial_qty.get(symbol)
        if start is None:
            priced = False
            assumption_failures.append(_rule("ASSUMPTION_QUOTED_PRICE", symbol, False, None, None))
            continue
        trade_cost += delta * price
        final_qty[symbol] = start + delta

    cash_before = _cash(scenario) + _flow_total(scenario, "DEPOSIT")
    withdrawals = _flow_total(scenario, "WITHDRAWAL")
    if priced:
        cash_after = cash_before - trade_cost
        # Scenarios without a withdrawal have no cash_before_withdrawal stage.
        # Final cash is the cash after trades, which starts from cash_before_trades.
        final_cash = cash_after - withdrawals if withdrawals else cash_after
    else:
        cash_after = None
        final_cash = None

    if closed_ledger and final_cash is not None and final_cash < -roundoff:
        assumption_failures.append(
            _rule("ASSUMPTION_NO_NEGATIVE_CASH", "CASH", False, final_cash, 0)
        )
    if closed_ledger:
        for symbol, quantity in final_qty.items():
            if quantity is not None and quantity < -roundoff:
                assumption_failures.append(
                    _rule("ASSUMPTION_NO_NEGATIVE_SHARES", symbol, False, quantity, 0)
                )

    final_values = {}
    if priced:
        for symbol, quantity in final_qty.items():
            price = prices.get(symbol)
            if quantity is None or price is None:
                continue
            final_values[symbol] = quantity * price
    holdings_value = sum(final_values.values()) if priced else None
    final_total = final_cash + holdings_value if priced and final_cash is not None else None
    allocations = _allocations(holdings, final_values, final_cash, final_total) if priced else {}

    metrics = {
        "initial_total": _optional_float(scenario["portfolio"].get("total_value"), None),
        "cash_before_trades": cash_before,
        "final_total": final_total,
        "final_cash": final_cash,
        "allocations": allocations if priced and final_total not in (None, 0) else None,
        "cash_before_withdrawal": cash_after if withdrawals and priced else None,
        "historical_total": _historical_total(scenario),
        "additional_liquidity_needed": _additional_liquidity(scenario, withdrawals),
        "direct_technology_pct": _direct_technology(scenario, final_values, final_total)
        if priced
        else None,
    }
    reports = _expected_reports(metrics)
    observed = {
        "initial_total": metrics["initial_total"],
        "cash_before_trades": cash_before,
        "cash_before_withdrawal": metrics["cash_before_withdrawal"],
        "final_cash": final_cash,
        "final_total": final_total,
        "allocations": allocations,
        "final_quantities": {symbol: qty for symbol, qty in final_qty.items() if qty is not None},
        "final_values": final_values,
        "direct_technology_pct": metrics["direct_technology_pct"],
        "historical_total": metrics["historical_total"],
        "additional_liquidity_needed": metrics["additional_liquidity_needed"],
    }
    return {
        "priced": priced,
        "roundoff": roundoff,
        "net": net,
        "holdings": holdings,
        "final_qty": final_qty,
        "final_values": final_values,
        "final_cash": final_cash,
        "final_total": final_total,
        "allocations": allocations,
        "cash_before_trades": cash_before,
        "metrics": metrics,
        "reports": reports,
        "observed": observed,
        "assumption_failures": assumption_failures,
        "unknown_net": unknown_net,
    }


def _expected_reports(metrics: dict) -> dict:
    reports = {}

    def add(subject, measure, stage, unit, value, kind):
        if value is None:
            return
        reports[(subject, measure, stage, "NONE")] = (unit, value, kind)

    add("PORTFOLIO", "VALUE", "INITIAL", "MONEY", metrics.get("initial_total"), "MONEY")
    add("PORTFOLIO", "VALUE", "FINAL", "MONEY", metrics.get("final_total"), "MONEY")
    add("PORTFOLIO", "VALUE", "HISTORICAL", "MONEY", metrics.get("historical_total"), "MONEY")
    add("PORTFOLIO", "DIFFERENCE", "FINAL", "MONEY", metrics.get("additional_liquidity_needed"), "MONEY")
    add("CASH", "VALUE", "BEFORE_TRADES", "MONEY", metrics.get("cash_before_trades"), "MONEY")
    add("CASH", "VALUE", "BEFORE_WITHDRAWAL", "MONEY", metrics.get("cash_before_withdrawal"), "MONEY")
    add("CASH", "VALUE", "FINAL", "MONEY", metrics.get("final_cash"), "MONEY")
    tech = metrics.get("direct_technology_pct")
    add("TECHNOLOGY", "ALLOCATION", "FINAL", "FRACTION", tech, "FRACTION")
    allocations = metrics.get("allocations") or {}
    for name, fraction in allocations.items():
        add(name, "ALLOCATION", "FINAL", "FRACTION", fraction, "FRACTION")
    return reports


def _allocations(holdings, final_values, final_cash, final_total) -> dict:
    if final_total in (None, 0) or final_cash is None:
        return {}
    by_class: dict[str, float] = {}
    cash_value = final_cash
    for symbol, value in final_values.items():
        asset_class = holdings.get(symbol, {}).get("asset_class")
        if asset_class == "CASH":
            cash_value += value
        elif asset_class:
            by_class[asset_class] = by_class.get(asset_class, 0.0) + value
    allocations = {name: value / final_total for name, value in by_class.items()}
    allocations["CASH"] = cash_value / final_total
    return allocations


def _direct_technology(scenario, final_values, final_total):
    if final_total in (None, 0):
        return None
    for restriction in scenario["investment_policy"].get("restrictions") or []:
        if restriction.get("restriction_type") != "MAX_DIRECT_SECTOR":
            continue
        symbols = restriction.get("parameters_json", {}).get("symbols") or []
        return sum(final_values.get(symbol, 0.0) for symbol in symbols) / final_total
    return None


def _historical_total(scenario):
    cash = _optional_float(scenario["portfolio"].get("cash_balance"), None)
    if cash is None:
        return None
    total = cash
    for holding in scenario["portfolio"]["holdings"]:
        quantity = _optional_float(holding.get("quantity"), None)
        price = _optional_float(holding.get("historical_price"), None)
        if quantity is None or price is None:
            return None
        total += quantity * price
    return total


def _additional_liquidity(scenario, withdrawals: float):
    if not withdrawals:
        return None
    minimum = None
    for restriction in scenario["investment_policy"].get("restrictions") or []:
        if restriction.get("restriction_type") == "MIN_CASH":
            minimum = _optional_float(restriction.get("parameters_json", {}).get("amount"), None)
    if minimum is None:
        return None
    opening = _cash(scenario)
    return max(0.0, withdrawals + minimum - opening)


def _restriction_rows(scenario: dict, ledger: dict, params: dict) -> list[dict]:
    roundoff = ledger["roundoff"]
    whole_share = _optional_float(params.get("whole_share_tolerance"), 0.0)
    target_tolerance = _optional_float(params.get("target_allocation_tolerance"), None)
    rows = []
    for restriction in scenario["investment_policy"].get("restrictions") or []:
        kind = restriction.get("restriction_type")
        payload = restriction.get("parameters_json") or {}
        rows.extend(_one_restriction(kind, payload, ledger, roundoff, whole_share))
    if target_tolerance is not None and ledger["priced"]:
        for name, target in (scenario["investment_policy"].get("target_allocations") or {}).items():
            observed = ledger["allocations"].get(name, 0.0)
            passed = abs(observed - float(target)) <= target_tolerance + roundoff
            rows.append(_rule("TARGET_ALLOCATION", name, passed, observed, float(target)))
    return rows


def _one_restriction(kind, payload, ledger, roundoff, whole_share) -> list[dict]:
    if kind == "NO_SELL":
        return [
            _rule(
                "NO_SELL",
                symbol,
                ledger["net"].get(symbol, 0.0) >= -roundoff,
                ledger["net"].get(symbol, 0.0),
                0,
            )
            for symbol in payload.get("symbols") or []
        ]
    if kind == "NO_SELL_CLASS":
        asset_class = payload.get("asset_class")
        rows = []
        for symbol, holding in ledger["holdings"].items():
            if holding.get("asset_class") != asset_class:
                continue
            delta = ledger["net"].get(symbol, 0.0)
            rows.append(_rule("NO_SELL_CLASS", symbol, delta >= -roundoff, delta, 0))
        return rows or [_rule("NO_SELL_CLASS", asset_class, True, 0, 0)]
    if kind == "MAX_INDIVIDUAL_STOCK":
        return _cap_rows(
            "MAX_INDIVIDUAL_STOCK",
            [
                symbol
                for symbol, holding in ledger["holdings"].items()
                if holding.get("asset_type") == "INDIVIDUAL_STOCK"
            ],
            payload.get("max_pct"),
            ledger,
            roundoff,
        )
    if kind == "MAX_DIRECT_SECTOR":
        return [
            _fraction_row(
                "MAX_DIRECT_SECTOR",
                payload.get("sector") or "PORTFOLIO",
                payload.get("symbols") or [],
                payload.get("max_pct"),
                ledger,
                roundoff,
            )
        ]
    if kind == "CLASS_RANGE":
        return [_class_range_row(payload, ledger, roundoff)]
    if kind == "MIN_CASH":
        amount = _optional_float(payload.get("amount"), None)
        cash = ledger["final_cash"]
        passed = cash is not None and amount is not None and cash + roundoff >= amount
        return [_rule("MIN_CASH", "CASH", passed, cash, amount)]
    if kind == "WHOLE_SHARES":
        traded = [(symbol, delta) for symbol, delta in ledger["net"].items() if abs(delta) > roundoff]
        if not traded:
            return [_rule("WHOLE_SHARES", "PORTFOLIO", True, 0, 0)]
        rows = []
        for symbol, delta in traded:
            nearest = round(delta)
            passed = abs(delta - nearest) <= whole_share
            rows.append(_rule("WHOLE_SHARES", symbol, passed, delta, nearest))
        return rows
    if kind == "EXISTING_ASSETS_ONLY":
        unknown = _unknown_asset_rows(ledger, roundoff)
        if not unknown:
            return [_rule("EXISTING_ASSETS_ONLY", "PORTFOLIO", True, 0, 0)]
        return unknown
    if kind == "NO_BORROWING":
        cash = ledger["final_cash"]
        passed = cash is not None and cash >= -roundoff
        return [_rule("NO_BORROWING", "CASH", passed, cash, 0)]
    if kind == "NO_EXTERNAL_DEPOSITS":
        # Replay never credits cash that the scenario did not list.
        return [_rule("NO_EXTERNAL_DEPOSITS", "CASH", True, 0, 0)]
    if kind == "NO_SHORT_SELLING":
        shorts = [
            (symbol, quantity)
            for symbol, quantity in ledger["final_qty"].items()
            if quantity is not None and quantity < -roundoff
        ]
        if not shorts:
            return [_rule("NO_SHORT_SELLING", "PORTFOLIO", True, 0, 0)]
        return [_rule("NO_SHORT_SELLING", symbol, False, quantity, 0) for symbol, quantity in shorts]
    if kind == "PROHIBITED_CLASS":
        return [
            _fraction_row(
                "PROHIBITED_CLASS",
                payload.get("asset_class") or "PORTFOLIO",
                [
                    symbol
                    for symbol, holding in ledger["holdings"].items()
                    if holding.get("asset_class") == payload.get("asset_class")
                ],
                payload.get("max_pct"),
                ledger,
                roundoff,
            )
        ]
    if kind == "FIXED_PRICES":
        return [_rule("FIXED_PRICES", "PORTFOLIO", ledger["priced"], None, 0)]
    if kind == "ZERO_TAXES_AND_FEES":
        return [_rule("ZERO_TAXES_AND_FEES", "PORTFOLIO", True, 0, 0)]
    return [_rule(kind or "UNKNOWN", "PORTFOLIO", False, None, None)]


def _cap_rows(rule, symbols, max_pct, ledger, roundoff) -> list[dict]:
    limit = _optional_float(max_pct, None)
    if not symbols:
        return [_rule(rule, "PORTFOLIO", True, 0, limit)]
    rows = []
    for symbol in symbols:
        observed = _weight(ledger, [symbol])
        passed = observed is not None and limit is not None and observed <= limit + roundoff
        rows.append(_rule(rule, symbol, passed, observed, limit))
    return rows


def _fraction_row(rule, scope, symbols, max_pct, ledger, roundoff) -> dict:
    limit = _optional_float(max_pct, None)
    observed = _weight(ledger, symbols)
    passed = observed is not None and limit is not None and observed <= limit + roundoff
    return _rule(rule, scope, passed, observed, limit)


def _class_range_row(payload, ledger, roundoff) -> dict:
    scope = payload.get("asset_class") or "PORTFOLIO"
    observed = _weight(
        ledger,
        [
            symbol
            for symbol, holding in ledger["holdings"].items()
            if holding.get("asset_class") == scope
        ],
    )
    if scope == "CASH" and ledger["final_total"] not in (None, 0) and ledger["final_cash"] is not None:
        observed = ledger["final_cash"] / ledger["final_total"]
    minimum = _optional_float(payload.get("min_pct"), None)
    maximum = _optional_float(payload.get("max_pct"), None)
    if observed is None:
        return _rule("CLASS_RANGE", scope, False, None, maximum if maximum is not None else minimum)
    below = minimum is not None and observed < minimum - roundoff
    above = maximum is not None and observed > maximum + roundoff
    if below:
        return _rule("CLASS_RANGE", scope, False, observed, minimum)
    if above:
        return _rule("CLASS_RANGE", scope, False, observed, maximum)
    return _rule("CLASS_RANGE", scope, True, observed, maximum if maximum is not None else minimum)


def _weight(ledger, symbols):
    total = ledger["final_total"]
    if not ledger["priced"] or total in (None, 0):
        return None
    return sum(ledger["final_values"].get(symbol, 0.0) for symbol in symbols) / total


def _unknown_asset_rows(ledger: dict, roundoff: float = 0.0) -> list[dict]:
    return [
        _rule("EXISTING_ASSETS_ONLY", symbol, False, delta, 0)
        for symbol, delta in ledger.get("unknown_net", {}).items()
        if abs(delta) > roundoff
    ]


def _rule(rule, scope, passed, observed, limit) -> dict:
    return {
        "rule": rule,
        "scope": scope,
        "passed": bool(passed),
        "observed": observed,
        "limit": limit,
    }


def _independent_analysis(scenario: dict, response: dict, mapping: dict) -> dict:
    routes = mapping.get("metric_routes", {})
    facts = _scenario_facts(scenario)
    index = _observation_index(response)
    expected = {}
    reported = {}
    for key in mapping.get("required_private_metrics", []):
        expected[key] = facts.get(key)
        if _uses_report(routes.get(key, {})):
            reported[key] = _reported_metric(key, scenario, response, index, facts)
        else:
            reported[key] = expected[key]
    return {"expected": expected, "reported": reported}


def _independent_check(spec: dict, analysis: dict) -> dict:
    fraction_tolerance = _optional_float(spec["params_json"].get("fraction_tolerance"), 0.0)
    missing = []
    not_computed = []
    mismatches = []
    for key, expected in analysis["expected"].items():
        reported = analysis["reported"].get(key)
        if reported is None:
            missing.append(key)
            continue
        if expected is None:
            not_computed.append(key)
            continue
        if not _metric_matches(key, reported, expected, fraction_tolerance):
            mismatches.append(
                {
                    "reason": "value_mismatch",
                    "metric": key,
                    "observed": reported,
                    "expected": expected,
                }
            )
    if missing:
        reason = "missing_observation"
    elif not_computed:
        reason = "rule_violation"
    elif mismatches:
        reason = "value_mismatch"
    else:
        reason = None
    details = {}
    if reason:
        details = {
            "reason": reason,
            "missing_observation": missing,
            "not_computed": not_computed,
            "items": mismatches,
        }
    return _check_row(spec, reason is None, analysis["expected"], details)


def _metric_matches(key: str, reported, expected, fraction_tolerance: float) -> bool:
    if isinstance(expected, str):
        return reported == expected
    if isinstance(expected, dict):
        if not isinstance(reported, dict) or set(reported) != set(expected):
            return False
        return all(
            abs(float(reported[name]) - float(expected[name])) <= fraction_tolerance
            for name in expected
        )
    if key == "violation_count":
        return reported == expected
    if key.endswith("_cents"):
        return abs(float(reported) - float(expected)) <= 1e-9
    if key.endswith("_pct"):
        return abs(float(reported) - float(expected)) <= fraction_tolerance
    return abs(float(reported) - float(expected)) <= MONEY_OBSERVATION_TOLERANCE


def _exact_violations(spec: dict, scenario: dict, response: dict) -> dict:
    expected_rows = _expected_violations(scenario)
    params = spec["params_json"]
    if expected_rows is None:
        return _check_row(spec, False, None, {"not_computed": ["violations"]})
    seen: dict[tuple, dict] = {}
    duplicates = []
    for row in response["violations"]:
        key = (row["kind"], row["scope"])
        if key in seen:
            duplicates.append({"kind": key[0], "scope": key[1]})
        else:
            seen[key] = row
    expected = {(row["kind"], row["scope"]): row for row in expected_rows}
    missing = [{"kind": kind, "scope": scope} for kind, scope in expected if (kind, scope) not in seen]
    extra = [{"kind": kind, "scope": scope} for kind, scope in seen if (kind, scope) not in expected]
    fraction_tolerance = _optional_float(params.get("fraction_tolerance"), 0.0)
    mismatches = []
    for key, wanted in expected.items():
        got = seen.get(key)
        if got is None:
            continue
        if (
            abs(float(got["observed"]) - float(wanted["observed"])) > fraction_tolerance
            or abs(float(got["limit"]) - float(wanted["limit"])) > fraction_tolerance
        ):
            mismatches.append(
                {
                    "reason": "value_mismatch",
                    "kind": key[0],
                    "scope": key[1],
                    "observed": got["observed"],
                    "expected": wanted["observed"],
                    "limit": got["limit"],
                    "expected_limit": wanted["limit"],
                }
            )
    crypto_count = params.get("crypto_count")
    crypto_problem = False
    if isinstance(crypto_count, int) and not isinstance(crypto_count, bool):
        found = sum(1 for row in expected_rows if row["kind"] == "PROHIBITED_CLASS")
        crypto_problem = found != crypto_count
    reject_extra = params.get("reject_extra_violations", True)
    unique = params.get("unique_rule_scope", True)
    extra_rows = extra if reject_extra else []
    duplicate_rows = duplicates if unique else []
    if missing:
        reason = "missed_violation"
    elif extra_rows or duplicate_rows or crypto_problem:
        reason = "extra_violation"
    elif mismatches:
        reason = "value_mismatch"
    else:
        reason = None
    details = {}
    if reason:
        details = {
            "reason": reason,
            "missing": missing,
            "extra": extra,
            "duplicates": duplicates,
            "items": mismatches,
        }
    return _check_row(spec, reason is None, expected_rows, details)


def _empty_trades(spec: dict, response: dict) -> dict:
    expected = spec["params_json"].get("expected", [])
    trades = response["trades"]
    passed = trades == expected
    details = {} if passed else {"reason": "rule_violation", "observed": trades, "expected": expected}
    return _check_row(spec, passed, trades, details)


def _scenario_facts(scenario: dict) -> dict:
    positions, cash, total = _position_values(scenario)
    facts: dict = {}
    if total is not None:
        facts["initial_total"] = total
    allocations = _class_allocations(positions, cash, total)
    if allocations is not None:
        facts["allocations"] = allocations
    violations = _expected_violations(scenario)
    if violations is not None:
        facts["violation_count"] = len(violations)
    facts.update(_fund_facts(scenario))
    facts.update(_retirement_facts(scenario))
    facts.update(_evidence_facts(scenario, positions, cash, total))
    return facts


def _position_values(scenario: dict):
    """Holding values from market_value. A null price is never a divisor or a factor."""
    positions = []
    for holding in scenario["portfolio"].get("holdings") or []:
        market_value = holding.get("market_value")
        if market_value is None:
            quantity = holding.get("quantity")
            price = holding.get("current_price")
            value = None if quantity is None or price is None else float(quantity) * float(price)
        else:
            value = float(market_value)
        positions.append(
            {
                "symbol": holding["symbol"],
                "asset_class": holding.get("asset_class"),
                "asset_type": holding.get("asset_type"),
                "value": value,
            }
        )
    cash = scenario["portfolio"].get("cash_balance")
    total = scenario["portfolio"].get("total_value")
    return (
        positions,
        None if cash is None else float(cash),
        None if total is None else float(total),
    )


def _class_allocations(positions, cash, total):
    if total in (None, 0) or any(position["value"] is None for position in positions):
        return None
    if cash is None and not positions:
        return None
    class_values: dict[str, float] = {}
    for position in positions:
        name = position["asset_class"] or "PORTFOLIO"
        class_values[name] = class_values.get(name, 0.0) + position["value"]
    if cash is not None:
        class_values["CASH"] = class_values.get("CASH", 0.0) + cash
    return {name: value / total for name, value in class_values.items()}


def _expected_violations(scenario: dict):
    positions, cash, total = _position_values(scenario)
    if total in (None, 0) or any(position["value"] is None for position in positions) or cash is None:
        return None
    class_values: dict[str, float] = {}
    for position in positions:
        name = position["asset_class"] or "PORTFOLIO"
        class_values[name] = class_values.get(name, 0.0) + position["value"]
    class_values["CASH"] = class_values.get("CASH", 0.0) + cash
    rows = []
    for restriction in scenario["investment_policy"].get("restrictions") or []:
        kind = restriction.get("restriction_type")
        payload = restriction.get("parameters_json") or {}
        if kind == "CLASS_RANGE":
            scope = payload.get("asset_class") or "PORTFOLIO"
            observed = class_values.get(scope, 0.0) / total
            minimum = _optional_float(payload.get("min_pct"), None)
            maximum = _optional_float(payload.get("max_pct"), None)
            if minimum is not None and observed < minimum - 1e-12:
                rows.append(_violation("CLASS_MIN", scope, observed, minimum))
            if maximum is not None and observed > maximum + 1e-12:
                rows.append(_violation("CLASS_MAX", scope, observed, maximum))
        elif kind == "MAX_INDIVIDUAL_STOCK":
            limit = _optional_float(payload.get("max_pct"), None)
            if limit is None:
                continue
            for position in positions:
                if position["asset_type"] != "INDIVIDUAL_STOCK":
                    continue
                observed = position["value"] / total
                if observed > limit + 1e-12:
                    rows.append(_violation("INDIVIDUAL_MAX", position["symbol"], observed, limit))
        elif kind == "PROHIBITED_CLASS":
            scope = payload.get("asset_class") or "PORTFOLIO"
            limit = _optional_float(payload.get("max_pct"), 0.0)
            observed = class_values.get(scope, 0.0) / total
            if observed > limit + 1e-12:
                rows.append(_violation("PROHIBITED_CLASS", scope, observed, limit))
    return rows


def _violation(kind: str, scope: str, observed: float, limit: float) -> dict:
    return {"kind": kind, "scope": scope, "observed": observed, "limit": limit}


def _fund_facts(scenario: dict) -> dict:
    options = scenario.get("investment_options") or []
    if not options:
        return {}
    finals = [(option["name"], _compound_option(option)) for option in options]
    facts = {_option_metric_key(name): value for name, value in finals}
    if len(finals) >= 2:
        facts["absolute_difference"] = abs(finals[0][1] - finals[1][1])
        facts["better_fund"] = max(finals, key=lambda item: item[1])[0]
    return facts


def _compound_option(option: dict) -> float:
    """Entry fee comes off first. Each year multiplies by (1 + gross) * (1 - fee)."""
    initial = float(option["initial_investment"])
    entry = float(option.get("entry_fee") or 0)
    growth = (1 + float(option["gross_annual_return"])) * (1 - float(option["annual_fee"]))
    return initial * (1 - entry) * growth ** int(option["years"])


def _option_metric_key(name: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in name.strip())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"{slug.strip('_')}_final"


def _retirement_facts(scenario: dict) -> dict:
    if scenario.get("calculation_type") != "RETIREMENT_PROJECTION":
        return {}
    assumptions = scenario.get("retirement_assumptions") or {}
    balance = _project_balance(scenario)
    if balance is None:
        return {}
    target = float(assumptions["target"])
    shortfall = target - balance
    rate = float(assumptions["monthly_interest_rate"])
    periods = int(assumptions["additional_contribution_duration_months"])
    payment = _annuity_payment(shortfall, rate, periods)
    return {
        "final_balance": balance,
        "target_shortfall": shortfall,
        "additional_monthly_contribution": payment,
        "additional_monthly_contribution_cents": _upward_cent(payment),
    }


def _project_balance(scenario: dict):
    """Interest for the month, then that month's end-of-month cash flows."""
    assumptions = scenario.get("retirement_assumptions") or {}
    total = scenario["portfolio"].get("total_value")
    if total is None or "months" not in assumptions or "monthly_interest_rate" not in assumptions:
        return None
    balance = float(total)
    rate = float(assumptions["monthly_interest_rate"])
    for month in range(1, int(assumptions["months"]) + 1):
        balance *= 1 + rate
        for flow in scenario.get("cash_flows") or []:
            if not _flow_covers(flow, month):
                continue
            amount = float(flow["amount"])
            if flow.get("flow_type") == "WITHDRAWAL":
                balance -= amount
            else:
                balance += amount
    return balance


def _flow_covers(flow: dict, month: int) -> bool:
    start = flow.get("start_month")
    duration = flow.get("duration_months")
    if start is None or duration is None:
        return False
    return int(start) <= month < int(start) + int(duration)


def _annuity_payment(shortfall: float, rate: float, periods: int) -> float:
    if periods <= 0:
        return shortfall
    if rate == 0:
        return shortfall / periods
    return shortfall * rate / ((1 + rate) ** periods - 1)


def _upward_cent(amount: float) -> float:
    scaled = amount * 100
    nearest = round(scaled)
    if abs(scaled - nearest) <= 1e-6:
        return nearest / 100
    return math.ceil(scaled - 1e-9) / 100


def _evidence_facts(scenario: dict, positions, cash, total) -> dict:
    facts: dict = {"locked_symbols": [], "restricted_symbols": [], "sellable_symbols": []}
    no_sell_symbols, no_sell_classes = _sale_blocks(scenario)
    locked = [
        position
        for position in positions
        if position["asset_class"] in no_sell_classes
        or (position["symbol"] in no_sell_symbols and position["asset_class"] == "STOCK")
    ]
    facts["locked_symbols"] = [position["symbol"] for position in locked]
    if locked and total not in (None, 0) and all(position["value"] is not None for position in locked):
        facts["locked_stock_value"] = sum(position["value"] for position in locked)
        facts["minimum_stock_pct"] = facts["locked_stock_value"] / total
    targets = scenario["investment_policy"].get("target_allocations") or {}
    tolerance = scenario["investment_policy"].get("allocation_tolerance")
    if "STOCK" in targets and isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool):
        facts["maximum_target_stock_pct"] = float(targets["STOCK"]) + float(tolerance)
        if "locked_stock_value" in facts and total not in (None, 0):
            facts["stock_value_excess_above_tolerance"] = (
                facts["locked_stock_value"] - total * facts["maximum_target_stock_pct"]
            )
    if no_sell_symbols:
        restricted = [position for position in positions if position["symbol"] in set(no_sell_symbols)]
        sellable = [position for position in positions if position["symbol"] not in set(no_sell_symbols)]
        facts["restricted_symbols"] = [position["symbol"] for position in restricted]
        facts["sellable_symbols"] = [position["symbol"] for position in sellable]
        if all(position["value"] is not None for position in restricted):
            facts["restricted_value"] = sum(position["value"] for position in restricted)
        if all(position["value"] is not None for position in sellable):
            facts["sellable_value"] = sum(position["value"] for position in sellable)
        if "sellable_value" in facts and cash is not None:
            facts["available_liquidity"] = cash + facts["sellable_value"]
        reserve = _min_cash_amount(scenario)
        if reserve is not None:
            facts["required_liquidity"] = _flow_total(scenario, "WITHDRAWAL") + reserve
            if "available_liquidity" in facts:
                facts["liquidity_shortfall"] = facts["required_liquidity"] - facts["available_liquidity"]
                facts["max_withdrawal_preserving_reserve"] = facts["available_liquidity"] - reserve
    return facts


def _sale_blocks(scenario: dict):
    symbols = []
    classes = []
    for restriction in scenario["investment_policy"].get("restrictions") or []:
        payload = restriction.get("parameters_json") or {}
        if restriction.get("restriction_type") == "NO_SELL":
            symbols.extend(payload.get("symbols") or [])
        elif restriction.get("restriction_type") == "NO_SELL_CLASS":
            asset_class = payload.get("asset_class")
            if asset_class:
                classes.append(asset_class)
    return symbols, classes


def _min_cash_amount(scenario: dict):
    for restriction in scenario["investment_policy"].get("restrictions") or []:
        if restriction.get("restriction_type") == "MIN_CASH":
            return _optional_float((restriction.get("parameters_json") or {}).get("amount"), None)
    return None


def _reported_metric(key: str, scenario: dict, response: dict, index: dict, facts: dict):
    if key == "violation_count":
        return len(response["violations"])
    if key == "better_fund":
        selection = response["metrics"].get("selection")
        return selection if isinstance(selection, str) and selection else None
    if key == "initial_total":
        return _lookup(index, "PORTFOLIO", "VALUE", "INITIAL", "NONE", "MONEY")
    if key == "allocations":
        return _reported_allocations(scenario, index)
    if key.endswith("_final"):
        subject = _option_subject(scenario, key)
        if subject is None:
            return None
        return _lookup(index, subject, "VALUE", "FINAL", "NONE", "MONEY")
    if key == "absolute_difference":
        return _lookup(index, "OPTIONS", "DIFFERENCE", "FINAL", "NONE", "MONEY")
    if key == "final_balance":
        return _lookup(index, "PORTFOLIO", "VALUE", "FINAL", "NONE", "MONEY")
    if key == "target_shortfall":
        return _lookup(index, "PORTFOLIO", "DIFFERENCE", "FINAL", "NONE", "MONEY")
    if key == "additional_monthly_contribution":
        return _lookup(index, "PORTFOLIO", "PAYMENT", "FINAL", "NONE", "MONEY")
    if key == "additional_monthly_contribution_cents":
        return _lookup(index, "PORTFOLIO", "PAYMENT", "FINAL", "UPWARD_CENT", "MONEY")
    if key == "locked_stock_value":
        return _sum_observations(index, facts.get("locked_symbols") or [])
    if key == "minimum_stock_pct":
        locked = _sum_observations(index, facts.get("locked_symbols") or [])
        portfolio = _lookup(index, "PORTFOLIO", "VALUE", "INITIAL", "NONE", "MONEY")
        if locked is None or portfolio in (None, 0):
            return None
        return locked / portfolio
    if key == "stock_value_excess_above_tolerance":
        locked = _sum_observations(index, facts.get("locked_symbols") or [])
        portfolio = _lookup(index, "PORTFOLIO", "VALUE", "INITIAL", "NONE", "MONEY")
        cap = facts.get("maximum_target_stock_pct")
        if locked is None or portfolio is None or cap is None:
            return None
        return locked - portfolio * cap
    if key == "restricted_value":
        return _sum_observations(index, facts.get("restricted_symbols") or [])
    if key == "sellable_value":
        return _sum_observations(index, facts.get("sellable_symbols") or [])
    if key == "available_liquidity":
        sellable = _sum_observations(index, facts.get("sellable_symbols") or [])
        cash = _lookup(index, "CASH", "VALUE", "INITIAL", "NONE", "MONEY")
        if sellable is None or cash is None:
            return None
        return cash + sellable
    if key == "liquidity_shortfall":
        available = _reported_metric("available_liquidity", scenario, response, index, facts)
        required = facts.get("required_liquidity")
        if available is None or required is None:
            return None
        return required - available
    if key == "max_withdrawal_preserving_reserve":
        available = _reported_metric("available_liquidity", scenario, response, index, facts)
        reserve = _min_cash_amount(scenario)
        if available is None or reserve is None:
            return None
        return available - reserve
    return None


def _option_subject(scenario: dict, key: str):
    for option in scenario.get("investment_options") or []:
        if _option_metric_key(option["name"]) == key:
            return option["name"]
    return None


def _reported_allocations(scenario: dict, index: dict):
    portfolio = _lookup(index, "PORTFOLIO", "VALUE", "INITIAL", "NONE", "MONEY")
    cash = _lookup(index, "CASH", "VALUE", "INITIAL", "NONE", "MONEY")
    if portfolio in (None, 0) or cash is None:
        return None
    class_values: dict[str, float] = {}
    for holding in scenario["portfolio"].get("holdings") or []:
        value = _lookup(index, holding["symbol"], "VALUE", "INITIAL", "NONE", "MONEY")
        if value is None:
            return None
        name = holding.get("asset_class") or "PORTFOLIO"
        class_values[name] = class_values.get(name, 0.0) + value
    allocations = {name: value / portfolio for name, value in class_values.items() if name != "CASH"}
    allocations["CASH"] = (cash + class_values.get("CASH", 0.0)) / portfolio
    return allocations


def _sum_observations(index: dict, symbols: list[str]):
    total = 0.0
    for symbol in symbols:
        value = _lookup(index, symbol, "VALUE", "INITIAL", "NONE", "MONEY")
        if value is None:
            return None
        total += value
    return total


def _observation_index(response: dict) -> dict:
    return {
        tuple(obs[field] for field in _OBSERVATION_FIELDS): obs
        for obs in response["metrics"]["observations"]
    }


def _lookup(index: dict, subject: str, measure: str, stage: str, rounding: str, unit: str):
    obs = index.get((subject, measure, stage, rounding))
    if obs is None or obs.get("unit") != unit:
        return None
    return float(obs["value"])


def _schema_errors(response: dict, scenario: dict, scenario_id: str) -> list[str]:
    schema = scenario["output_requirements"]["response_schema"]
    errors = _validate(response, schema, "$")
    if errors:
        return errors
    seen = set()
    for index, obs in enumerate(response["metrics"]["observations"]):
        key = tuple(obs[field] for field in _OBSERVATION_FIELDS)
        if key in seen:
            errors.append(f"$.metrics.observations[{index}] repeats {key}")
        seen.add(key)
    if response["scenario_id"] != scenario_id:
        errors.append("scenario_id does not match the scenario being scored")
    if response["answer_type"] != scenario["answer_type"]:
        errors.append("answer_type does not match the scenario")
    return errors


def _validate(instance, schema: dict, path: str) -> list[str]:
    if "oneOf" in schema:
        if any(not _validate(instance, option, path) for option in schema["oneOf"]):
            return []
        return [f"{path} does not match any allowed schema"]
    expected = schema.get("type")
    if expected is not None and not _matches_type(instance, expected):
        return [f"{path} must be {expected}"]
    if "enum" in schema and instance not in schema["enum"]:
        return [f"{path} must be one of {schema['enum']}"]
    errors = []
    if expected == "string" and len(instance) < schema.get("minLength", 0):
        errors.append(f"{path} is too short")
    if expected == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}.{key} is required")
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in properties:
                    errors.append(f"{path}.{key} is not allowed")
        for key, subschema in properties.items():
            if key in instance:
                errors.extend(_validate(instance[key], subschema, f"{path}.{key}"))
    if expected == "array" and "items" in schema:
        for index, item in enumerate(instance):
            errors.extend(_validate(item, schema["items"], f"{path}[{index}]"))
    return errors


def _matches_type(instance, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "null":
        return instance is None
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool) and math.isfinite(instance)
    return False


def _parse_response(response: dict | str):
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError as exc:
            return None, f"response is not valid JSON ({exc.msg})"
    if isinstance(response, bool) or not isinstance(response, dict):
        return None, "response must be a JSON object"
    return response, None


def _cash(scenario) -> float:
    return float(scenario["portfolio"]["cash_balance"])


def _flow_total(scenario, flow_type: str) -> float:
    total = 0.0
    for flow in scenario.get("cash_flows") or []:
        if flow.get("flow_type") == flow_type:
            total += float(flow["amount"])
    return total


def _spec(checks: list[dict], check_type: str) -> dict | None:
    for item in checks:
        if item["check_type"] == check_type:
            return item
    return None


def _optional_float(value, default):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
