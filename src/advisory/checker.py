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

# Reported fractions in the files use 1e-6, which rejects a figure rounded to
# three decimals. Compare those reports at 5e-4. Replayed limits stay strict.
REPORTED_FRACTION_TOLERANCE = 5e-4
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
    replay_params = replay_spec["params_json"] if replay_spec else {}
    ledger = _replay(scenario, parsed["trades"], replay_params)
    results = []
    restriction_results: list[dict] = []
    for spec in checks:
        kind = spec["check_type"]
        if kind == "SCHEMA":
            results.append(_check_row(spec, True, None, {}))
        elif kind == "REQUIRED_METRICS":
            results.append(_required_metrics(spec, mapping, ledger))
        elif kind == "STATUS":
            results.append(_status(spec, parsed))
        elif kind == "REPLAY_TRADES":
            rows, mismatches = _judge_replay(scenario, parsed, ledger, spec["params_json"])
            restriction_results = rows
            results.append(_replay_check(spec, ledger, rows, mismatches))
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


def _required_metrics(spec: dict, mapping: dict, ledger: dict) -> dict:
    routes = mapping.get("metric_routes", {})
    missing_observation = []
    not_computed = []
    observed = {}
    for key in spec["params_json"].get("keys", []):
        source = routes.get(key, {}).get("source", "REPLAY_DERIVED")
        if source.startswith("REPORTED"):
            missing_observation.append(key)
            continue
        value = ledger["metrics"].get(key)
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
    rows.extend(ledger["assumption_failures"])
    mismatches = _reported_mismatches(response, ledger, params)
    return rows, mismatches


def _reported_mismatches(response: dict, ledger: dict, params: dict) -> list[dict]:
    """Observations are optional on a trade plan. Present ones must match."""
    money_tolerance = _optional_float(params.get("money_tolerance"), MONEY_OBSERVATION_TOLERANCE)
    expected = ledger["reports"]
    mismatches = []
    for obs in response["metrics"]["observations"]:
        key = tuple(obs[field] for field in _OBSERVATION_FIELDS)
        if key not in expected:
            continue
        unit, value, kind = expected[key]
        tolerance = REPORTED_FRACTION_TOLERANCE if kind == "FRACTION" else money_tolerance
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
    for trade in trades:
        symbol = trade["symbol"]
        net[symbol] = net.get(symbol, 0.0) + float(trade["quantity_delta"])

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
        unknown = [
            symbol
            for symbol, delta in ledger["net"].items()
            if abs(delta) > roundoff and symbol not in ledger["holdings"]
        ]
        if not unknown:
            return [_rule("EXISTING_ASSETS_ONLY", "PORTFOLIO", True, 0, 0)]
        return [
            _rule("EXISTING_ASSETS_ONLY", symbol, False, ledger["net"][symbol], 0) for symbol in unknown
        ]
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


def _rule(rule, scope, passed, observed, limit) -> dict:
    return {
        "rule": rule,
        "scope": scope,
        "passed": bool(passed),
        "observed": observed,
        "limit": limit,
    }


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
