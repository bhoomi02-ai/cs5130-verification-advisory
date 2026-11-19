import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

# --------------------------------------------------------------------------
# Paths and constants. Every path is anchored to this file, never to the
# current working directory, so the guard below cannot be bypassed by running
# the script from another folder.
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "generated_variants"
STAGING_DIR = ROOT / "staging"
CONSOLIDATED_DIR = ROOT / "consolidated"
SPECS_FILE = ROOT / "variant_specs.json"
MANIFEST_FILE = ROOT / "approved_manifest.json"
BASE_DIRS = (ROOT / "data" / "scenarios", ROOT)

APPROVED_RE = re.compile(r"T0[1-8]-v00[1-4](_ground_truth|_report)?\.json")
VARIANT_RE = re.compile(r"T0[1-8]-v\d{3}")
EXPECTED_ANSWER_TYPE = {
    "S01": "TRADE_PLAN", "S02": "TRADE_PLAN", "S03": "VIOLATION_LIST", "S04": "NUMERIC_RESULT",
    "S05": "TRADE_PLAN", "S06": "NUMERIC_RESULT", "S07": "TRADE_PLAN", "S08": "TRADE_PLAN",
}
CONTRACT_ID = "NEUTRAL_OUTPUT_V1"
_VARIANT_ID_RE = re.compile(r"T(\d{2})-v\d{3}")

EPS = 1e-9          # constraint round-off
CASH_TOL = 0.01     # money tolerance for cash checks
TOTAL_TOL = 0.02    # portfolio total consistency
MAX_TARGET_TOL = 0.0049  # solver stays strictly inside the stated +/-0.5pp band


# --------------------------------------------------------------------------
# Safe I/O
# --------------------------------------------------------------------------
def is_approved(path):
    """True for the approved variant files (scenario, ground truth, report)."""
    p = Path(path).resolve()
    return p.parent == DATA_DIR.resolve() and APPROVED_RE.fullmatch(p.name) is not None


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data, *, overwrite=False):
    """Atomically write JSON. Approved files are never writable; existing files
    are only replaced when overwrite=True."""
    path = Path(path)
    if is_approved(path):
        raise PermissionError(f"BLOCKED: approved dataset file is read-only: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def approved_files():
    return sorted(p for p in DATA_DIR.glob("*.json") if APPROVED_RE.fullmatch(p.name))


def manifest_create():
    """Record a SHA-256 for every approved file. Never replaces an existing manifest."""
    files = approved_files()
    if not files:
        raise FileNotFoundError(f"No approved files in {DATA_DIR}")
    write_json(MANIFEST_FILE, {p.name: sha256(p) for p in files})
    print(f"Manifest written: {len(files)} files -> {MANIFEST_FILE.name}")


def manifest_check():
    """Report approved files that were modified, deleted or are not in the manifest."""
    recorded = read_json(MANIFEST_FILE)
    current = {p.name: sha256(p) for p in approved_files()}
    modified = sorted(n for n in recorded if n in current and recorded[n] != current[n])
    missing = sorted(set(recorded) - set(current))
    unlisted = sorted(set(current) - set(recorded))
    for label, names in (("MODIFIED", modified), ("MISSING", missing), ("UNLISTED", unlisted)):
        for name in names:
            print(f"{label}: {name}")
    ok = not (modified or missing)
    print(f"MANIFEST CHECK: {'PASS' if ok else 'FAIL'} ({len(recorded)} recorded files)")
    return ok


def base_scenario_id(scenario_id):
    """The hand-made scenario a variant comes from: T03-v002 -> S03 (an S-id maps to itself)."""
    match = _VARIANT_ID_RE.fullmatch(scenario_id)
    return "S" + match.group(1) if match else scenario_id


def identity(record):
    """Id of a scenario / ground-truth record. Reads the old two-field form (scenario_id=S01 + variant_id) too."""
    return record.get("variant_id") or record["scenario_id"]


def load_base(scenario_id):
    for folder in BASE_DIRS:
        path = folder / f"{scenario_id}.json"
        if path.is_file():
            return read_json(path)
    raise FileNotFoundError(f"Base scenario {scenario_id}.json not found in {BASE_DIRS}")


def load_specs():
    return read_json(SPECS_FILE)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def rules_of(policy):
    """Group restriction parameters by restriction_type."""
    grouped = defaultdict(list)
    for rule in (policy or {}).get("restrictions", []):
        grouped[rule["restriction_type"]].append(rule.get("parameters_json") or {})
    return grouped


def split_cash_flows(scenario):
    """Return (deposits, withdrawals, errors) from the scenario's cash flows."""
    deposits = withdrawals = 0.0
    errors = []
    for flow in scenario.get("cash_flows", []):
        amount = float(flow["amount"])
        if not math.isfinite(amount) or amount < 0:
            errors.append("Invalid cash flow amount")
        elif flow["flow_type"] == "DEPOSIT":
            deposits += amount
        elif flow["flow_type"] == "WITHDRAWAL":
            withdrawals += amount
        else:
            errors.append(f"Unsupported cash flow: {flow['flow_type']}")
    return deposits, withdrawals, errors


def holdings_value(holdings):
    return sum(float(h["quantity"]) * float(h["current_price"]) for h in holdings)


def class_values(holdings, cash):
    values = defaultdict(float)
    for h in holdings:
        values[h["asset_class"]] += float(h["quantity"]) * float(h["current_price"])
    values["CASH"] += cash
    return values


# Restrictions the replay verifier understands. The last three are enforced
# structurally (only existing symbols, no negative cash, quoted prices).
CHECKED_RULES = {
    "NO_SELL", "WHOLE_SHARES", "NO_EXTERNAL_DEPOSITS", "MIN_CASH", "MAX_INDIVIDUAL_STOCK",
    "MAX_DIRECT_SECTOR", "CLASS_RANGE", "PROHIBITED_CLASS",
}
STRUCTURAL_RULES = {"EXISTING_ASSETS_ONLY", "NO_BORROWING", "CURRENT_PRICES_ONLY"}


# --------------------------------------------------------------------------
# Trade-plan replay (verifier)
# --------------------------------------------------------------------------
def verify_trade_plan(scenario, trades):
    """Replay `trades` on the scenario and check every restriction.

    Deposits are added before trading, withdrawals are taken after trading.
    Returns a dict with `valid`, `errors` and the final balances.
    """
    portfolio = scenario["portfolio"]
    policy = scenario.get("investment_policy", {})
    rules = rules_of(policy)
    holdings = {h["symbol"]: dict(h) for h in portfolio["holdings"]}
    initial_cash = float(portfolio["cash_balance"])
    deposits, withdrawals, errors = split_cash_flows(scenario)

    for kind in rules:
        if kind not in CHECKED_RULES | STRUCTURAL_RULES:
            errors.append(f"Unsupported restriction: {kind}")
    if rules["NO_EXTERNAL_DEPOSITS"] and deposits > 0:
        errors.append("NO_EXTERNAL_DEPOSITS violated")

    no_sell = {s for params in rules["NO_SELL"] for s in params.get("symbols", [])}
    net = defaultdict(float)
    for trade in trades:
        symbol, delta = trade.get("symbol"), trade.get("quantity_delta")
        if symbol not in holdings:
            errors.append(f"Unknown asset: {symbol}")
        elif not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta == 0:
            errors.append(f"Invalid trade quantity: {symbol}")
        else:
            if symbol in no_sell and delta < 0:
                errors.append(f"NO_SELL violated: {symbol}")  # a later buy must not cancel it
            net[symbol] += delta

    if rules["WHOLE_SHARES"]:
        errors += [f"WHOLE_SHARES violated: {s}" for s, d in net.items() if abs(d - round(d)) > EPS]

    cash = initial_cash + deposits
    for symbol, delta in net.items():
        asset = holdings[symbol]
        new_quantity = float(asset["quantity"]) + delta
        price = float(asset["current_price"])
        if new_quantity < -EPS:
            errors.append(f"Insufficient shares: {symbol}")
        elif price <= 0:
            errors.append(f"Invalid price: {symbol}")
        else:
            cash -= delta * price
            asset["quantity"] = max(0.0, new_quantity)

    cash_before_withdrawal = cash
    cash -= withdrawals
    if cash_before_withdrawal < -CASH_TOL:
        errors.append("Insufficient cash before withdrawal")
    if cash < -CASH_TOL:
        errors.append(f"Negative final cash: {cash:.2f}")
    for params in rules["MIN_CASH"]:
        if cash + CASH_TOL < params["amount"]:
            errors.append(f"MIN_CASH violated: {cash:.2f} < {params['amount']:.2f}")

    final_holdings = list(holdings.values())
    for h in final_holdings:
        h["market_value"] = round(float(h["quantity"]) * float(h["current_price"]), 2)
    values = class_values(final_holdings, cash)
    final_total = holdings_value(final_holdings) + cash
    stocks = {h["symbol"]: h["quantity"] * h["current_price"]
              for h in final_holdings if h["asset_type"] == "INDIVIDUAL_STOCK"}

    allocations = {}
    if final_total <= 0:
        errors.append("Invalid final portfolio value")
    else:
        allocations = {c: v / final_total for c, v in values.items()}
        tolerance = policy.get("allocation_tolerance", 0)
        for asset_class, target in policy.get("target_allocations", {}).items():
            actual = allocations.get(asset_class, 0)
            if abs(actual - target) > tolerance + EPS:
                errors.append(f"TARGET_ALLOCATION violated: {asset_class} = {actual:.4f}, target = {target:.4f}")
        for params in rules["MAX_INDIVIDUAL_STOCK"]:
            for symbol, value in stocks.items():
                if value / final_total > params["max_pct"] + EPS:
                    errors.append(f"MAX_INDIVIDUAL_STOCK violated: {symbol} = {value / final_total:.4f}")
        for params in rules["MAX_DIRECT_SECTOR"]:
            exposure = sum(stocks.get(s, 0) for s in params.get("symbols", [])) / final_total
            if exposure > params["max_pct"] + EPS:
                errors.append(f"MAX_DIRECT_SECTOR violated: {params['sector']} = {exposure:.4f}")
        for params in rules["CLASS_RANGE"]:
            actual = allocations.get(params["asset_class"], 0)
            if actual < params["min_pct"] - EPS:
                errors.append(f"CLASS_RANGE minimum violated: {params['asset_class']}")
            if actual > params["max_pct"] + EPS:
                errors.append(f"CLASS_RANGE maximum violated: {params['asset_class']}")
        for params in rules["PROHIBITED_CLASS"]:
            if allocations.get(params["asset_class"], 0) > params.get("max_pct", 0) + EPS:
                errors.append(f"PROHIBITED_CLASS violated: {params['asset_class']}")

    return {
        "valid": not errors,
        "scenario_id": scenario["scenario_id"],
        "initial_cash": round(initial_cash, 2),
        "deposits": round(deposits, 2),
        "withdrawals": round(withdrawals, 2),
        "cash_before_withdrawal": round(cash_before_withdrawal, 2),
        "final_cash": round(cash, 2),
        "final_portfolio_value": round(final_total, 2),
        "allocations": {c: round(v, 6) for c, v in allocations.items()},
        "final_holdings": final_holdings,
        "errors": errors,
    }


# --------------------------------------------------------------------------
# Trade-plan solver (one MILP for S01, S02, S05, S07, S08)
# --------------------------------------------------------------------------
def solve_trade_plan(scenario, time_limit=60, whole_shares=None):
    """Find a feasible plan that minimises dollar turnover.

    whole_shares=None follows the WHOLE_SHARES restriction; the dataset builder and
    audit pass True because every client request asks for whole-share trades.

    Variables are the signed share change y_i and its absolute value u_i.
    Targets and caps are measured on the portfolio value after deposits and
    withdrawals; final cash must be at least MIN_CASH (or zero).
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    portfolio = scenario["portfolio"]
    policy = scenario["investment_policy"]
    rules = rules_of(policy)
    unknown = set(rules) - CHECKED_RULES - STRUCTURAL_RULES
    if unknown:
        raise ValueError(f"Unsupported restrictions: {sorted(unknown)}")

    holdings = portfolio["holdings"]
    symbols = [h["symbol"] for h in holdings]
    if not holdings or len(set(symbols)) != len(symbols):
        raise ValueError("Portfolio must have unique holdings")
    n = len(holdings)
    price = np.array([float(h["current_price"]) for h in holdings])
    qty = np.array([float(h["quantity"]) for h in holdings])
    cash = float(portfolio["cash_balance"])
    if np.any(price <= 0) or np.any(qty < 0) or cash < 0:
        raise ValueError("Invalid prices, quantities or cash")
    if abs(price @ qty + cash - float(portfolio["total_value"])) > TOTAL_TOL:
        raise ValueError("Portfolio total does not match holdings plus cash")

    deposits, withdrawals, flow_errors = split_cash_flows(scenario)
    if flow_errors:
        raise ValueError("; ".join(flow_errors))
    total = float(price @ qty) + cash + deposits - withdrawals
    if total <= 0:
        return {"found": False, "trades": [], "message": "Withdrawal exhausts the portfolio"}
    cash_after_flows = cash + deposits - withdrawals
    min_cash = max([0.0] + [float(p["amount"]) for p in rules["MIN_CASH"]])
    no_sell = {s for p in rules["NO_SELL"] for s in p.get("symbols", [])}
    if not no_sell <= set(symbols):
        raise ValueError("NO_SELL references an unknown symbol")
    whole_shares = bool(rules["WHOLE_SHARES"]) if whole_shares is None else whole_shares

    lower = np.concatenate([np.where([s in no_sell for s in symbols], 0.0, -qty), np.zeros(n)])
    upper = np.full(2 * n, np.inf)
    objective = np.concatenate([np.zeros(n), price])
    rows, lo, hi = [], [], []

    def constrain(row, low, high):
        rows.append(row)
        lo.append(low)
        hi.append(high)

    def spend_row(mask=None):  # coefficients of y for the value of (a subset of) holdings
        row = np.zeros(2 * n)
        for i in range(n):
            if mask is None or mask[i]:
                row[i] = price[i]
        return row

    for i in range(n):  # u_i >= |y_i|
        for sign in (1, -1):
            row = np.zeros(2 * n)
            row[i], row[n + i] = sign, -1
            constrain(row, -np.inf, 0)

    caps = [float(p["max_pct"]) for p in rules["MAX_INDIVIDUAL_STOCK"]]
    for i, h in enumerate(holdings):
        if caps and h["asset_type"] == "INDIVIDUAL_STOCK":
            constrain(spend_row([j == i for j in range(n)]), -np.inf, min(caps) * total - qty[i] * price[i])
    for p in rules["MAX_DIRECT_SECTOR"]:
        mask = [s in set(p["symbols"]) for s in symbols]
        start = sum(qty[i] * price[i] for i in range(n) if mask[i])
        constrain(spend_row(mask), -np.inf, float(p["max_pct"]) * total - start)

    final_cash_row = -spend_row()  # final cash = cash_after_flows + final_cash_row @ y
    constrain(final_cash_row, min_cash - cash_after_flows, np.inf)

    targets = policy.get("target_allocations", {})
    base_tolerance = float(policy.get("allocation_tolerance", 0))
    for tolerance in dict.fromkeys([min(base_tolerance, MAX_TARGET_TOL), base_tolerance]):
        class_rows = list(rows), list(lo), list(hi)
        for asset_class, target in targets.items():
            if asset_class == "CASH":
                row, start = final_cash_row, cash_after_flows
            else:
                mask = [h["asset_class"] == asset_class for h in holdings]
                row = spend_row(mask)
                start = sum(qty[i] * price[i] for i in range(n) if mask[i])
            constrain(row, (target - tolerance) * total - start, (target + tolerance) * total - start)
        result = milp(
            c=objective,
            integrality=np.concatenate([np.ones(n) if whole_shares else np.zeros(n), np.zeros(n)]),
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(np.array(rows), np.array(lo), np.array(hi)),
            options={"time_limit": time_limit},
        )
        if result.x is not None:
            break
        rows[:], lo[:], hi[:] = class_rows  # retry with the full stated tolerance

    if result.x is None:
        return {"found": False, "trades": [], "message": str(result.message)}
    trades = []
    for symbol, delta in zip(symbols, result.x[:n]):
        delta = int(round(delta)) if whole_shares else float(delta)
        if abs(delta) > 1e-7:
            trades.append({"symbol": symbol, "quantity_delta": delta})
    verification = verify_trade_plan(scenario, trades)
    return {"found": verification["valid"], "trades": trades, "verification": verification,
            "message": str(result.message)}


# --------------------------------------------------------------------------
# Policy-violation audit (VIOLATION_LIST scenarios)
# --------------------------------------------------------------------------
def policy_violations(scenario, digits=8):
    """Current policy violations, one per (kind, scope)."""
    portfolio = scenario["portfolio"]
    total = float(portfolio["total_value"])
    values = class_values(portfolio["holdings"], float(portfolio["cash_balance"]))
    found = {}

    def report(kind, scope, observed, limit):
        found.setdefault((kind, scope), {"kind": kind, "scope": scope,
                                         "observed": round(observed, digits), "limit": float(limit)})

    for kind, params_list in rules_of(scenario["investment_policy"]).items():
        for p in params_list:
            if kind == "CLASS_RANGE":
                share = values.get(p["asset_class"], 0) / total
                if share < p["min_pct"] - EPS:
                    report("CLASS_MIN", p["asset_class"], share, p["min_pct"])
                if share > p["max_pct"] + EPS:
                    report("CLASS_MAX", p["asset_class"], share, p["max_pct"])
            elif kind == "PROHIBITED_CLASS":
                share = values.get(p["asset_class"], 0) / total
                if share > p.get("max_pct", 0) + EPS:
                    report("PROHIBITED_CLASS", p["asset_class"], share, p.get("max_pct", 0))
            elif kind == "MAX_INDIVIDUAL_STOCK":
                for h in portfolio["holdings"]:
                    share = float(h["quantity"]) * float(h["current_price"]) / total
                    if h["asset_type"] == "INDIVIDUAL_STOCK" and share > p["max_pct"] + EPS:
                        report("INDIVIDUAL_MAX", h["symbol"], share, p["max_pct"])
            else:
                raise ValueError(f"Unsupported restriction in violation audit: {kind}")
    return list(found.values())


# --------------------------------------------------------------------------
# Numeric ground truth (S04 fund comparison, S06 retirement projection)
# --------------------------------------------------------------------------
def fund_comparison(scenario):
    """Final value of each fund: entry fee first, then (1+return)*(1-fee) per year."""
    options = scenario["investment_options"]
    if len(options) != 2:
        raise ValueError("Fund comparison needs exactly two funds")
    values = {}
    for o in options:
        if o["initial_investment"] <= 0 or o["years"] <= 0:
            raise ValueError("Invalid principal or horizon")
        if not (0 <= o["entry_fee"] < 1 and 0 <= o["annual_fee"] < 1 and o["gross_annual_return"] > -1):
            raise ValueError(f"Invalid fee or return for {o['name']}")
        growth = (1 + o["gross_annual_return"]) * (1 - o["annual_fee"])
        values[o["name"]] = o["initial_investment"] * (1 - o["entry_fee"]) * growth ** o["years"]
    difference = abs(values[options[0]["name"]] - values[options[1]["name"]])
    winner = None if difference < 1e-8 else max(values, key=values.get)
    return {"values": values, "difference": difference, "winner": winner}


def project_balance(initial, rate, months, contributions, withdrawals, extra=0.0):
    """Month by month: growth first, then end-of-month flows.

    contributions: (start_month, duration_months, amount); withdrawals: (month, amount).
    """
    balance = float(initial)
    for month in range(1, months + 1):
        balance *= 1 + rate
        balance += sum(a for s, d, a in contributions if s <= month < s + d) + extra
        balance -= sum(a for m, a in withdrawals if m == month)
    return balance


def retirement_plan(scenario):
    """Projection inputs read from the scenario itself (no separate config)."""
    ra = scenario["retirement_assumptions"]
    flows = scenario["cash_flows"]
    contributions = [(f["start_month"], f["duration_months"], f["amount"])
                     for f in flows if f["flow_type"] == "CONTRIBUTION"]
    withdrawals = [(f["start_month"], f["amount"]) for f in flows if f["flow_type"] == "WITHDRAWAL"]
    return (scenario["portfolio"]["total_value"], ra["monthly_interest_rate"], ra["months"],
            contributions, withdrawals, ra["target"])


def retirement_truth(scenario):
    """Projected balance, target gap and the smallest fixed extra monthly payment."""
    initial, rate, months, contributions, withdrawals, target = retirement_plan(scenario)
    baseline = project_balance(initial, rate, months, contributions, withdrawals)
    gap = target - baseline
    per_dollar = project_balance(initial, rate, months, contributions, withdrawals, 1.0) - baseline
    if per_dollar <= 0:
        raise ValueError("Extra contribution has no effect on the balance")
    exact = max(0.0, gap) / per_dollar
    rounded = float(Decimal(str(exact)).quantize(Decimal("0.01"), rounding=ROUND_CEILING))
    reached = project_balance(initial, rate, months, contributions, withdrawals, rounded)
    if reached + 1e-4 < target:
        raise AssertionError("Rounded extra contribution does not reach the target")
    return {"projected_balance": baseline, "target_minus_projected": gap,
            "extra_monthly_exact": exact, "extra_monthly_rounded_up": rounded,
            "balance_with_extra": reached}


def numeric_ground_truth(scenario):
    """Rounded record stored in the per-variant ground-truth file."""
    kind = scenario.get("calculation_type")
    if kind == "FUND_COMPARISON":
        r = fund_comparison(scenario)
        return {"status": "FEASIBLE", "trades": [],
                "fund_values": {n: round(v, 2) for n, v in r["values"].items()},
                "absolute_difference": round(r["difference"], 2), "winner": r["winner"]}
    if kind == "RETIREMENT_PROJECTION":
        r = retirement_truth(scenario)
        return {"status": "FEASIBLE", "trades": [],
                "projected_balance": round(r["projected_balance"], 2),
                "target_balance": round(scenario["retirement_assumptions"]["target"], 2),
                "shortfall": round(max(0.0, r["target_minus_projected"]), 2),
                "extra_monthly_exact": round(r["extra_monthly_exact"], 6),
                "extra_monthly_rounded_up": r["extra_monthly_rounded_up"],
                "verified_final_balance": round(r["balance_with_extra"], 2),
                "target_reached": True}
    raise ValueError(f"Unsupported calculation_type: {kind}")


# --------------------------------------------------------------------------
# Variant builders (data comes from variant_specs.json)
# --------------------------------------------------------------------------
# Order in which each scenario's managed restrictions are appended.
RULE_ORDER = {
    "S01": ("NO_SELL", "MAX_INDIVIDUAL_STOCK"),
    "S02": ("MIN_CASH", "NO_SELL"),
    "S05": ("NO_SELL", "MAX_INDIVIDUAL_STOCK", "MAX_DIRECT_SECTOR"),
    "S07": ("NO_SELL", "MAX_INDIVIDUAL_STOCK"),
    "S08": ("NO_SELL", "MIN_CASH"),
}
FLOW_FIELD = {"S02": ("WITHDRAWAL", "withdrawal"), "S07": ("DEPOSIT", "deposit"), "S08": ("WITHDRAWAL", "withdrawal")}
TECH_SYMBOLS = ["AAPL", "MSFT", "NVDA"]


def _managed_rule(kind, spec):
    if kind == "NO_SELL":
        return {"symbols": spec["no_sell"]} if spec["no_sell"] else None
    if kind == "MAX_INDIVIDUAL_STOCK":
        return {"max_pct": spec["individual_cap"]}
    if kind == "MAX_DIRECT_SECTOR":
        return {"sector": "TECHNOLOGY", "symbols": TECH_SYMBOLS, "max_pct": spec["sector_cap"]}
    return {"amount": float(spec["min_cash"])}  # MIN_CASH


def _request_text(sid, spec):
    """Client request for a portfolio variant, written from the spec values."""
    protected = ", ".join(spec.get("no_sell", [])) or "None"
    t = spec.get("targets", {})
    if sid == "S01":
        return (f"My financial objective is {spec['story']}. Rebalance my portfolio to {t['STOCK']:.0%} stocks, "
                f"{t['BOND']:.0%} bonds, and {t['CASH']:.0%} cash. Do not sell {protected}. "
                f"No individual stock may exceed {spec['individual_cap']:.0%} of the total portfolio value. "
                "Use whole-share trades only. Do not use borrowing or external deposits. Provide a valid trade plan.")
    if sid == "S02":
        return (f"{spec['story']} Withdraw ${spec['withdrawal']:,.0f} from my existing portfolio. "
                f"After completing the withdrawal, maintain at least ${spec['min_cash']:,.0f} in cash. "
                f"Assets prohibited from sale: {protected}. Use whole-share trades and existing portfolio assets only. "
                "Do not borrow or add external deposits. Provide an executable trade plan.")
    if sid == "S05":
        return (f"My financial objective is {spec['story']}. Rebalance my existing portfolio to {t['STOCK']:.0%} stocks, "
                f"{t['BOND']:.0%} bonds, and {t['CASH']:.0%} cash. Do not sell {protected}. "
                f"No individual stock may exceed {spec['individual_cap']:.0%} of the total portfolio value. "
                f"The combined direct technology exposure of AAPL, MSFT and NVDA must not exceed {spec['sector_cap']:.0%}. "
                "Use existing assets and whole-share trades only. Do not borrow or add external deposits. "
                "Provide a valid trade plan.")
    if sid == "S07":
        return (f"My financial objective is {spec['story']}. I will deposit ${spec['deposit']:,.0f} into my existing "
                f"portfolio before rebalancing. After the deposit, I want {t['STOCK']:.0%} in stocks, "
                f"{t['BOND']:.0%} in bonds, and {t['CASH']:.0%} in cash. Do not sell {protected}. "
                f"No individual stock may exceed {spec['individual_cap']:.0%} of the portfolio value after deposit. "
                "Use current prices, whole-share trades, and existing securities only. "
                "Do not borrow or make any additional external deposits. Provide a valid trade plan.")
    if sid == "S08":
        return (f"My objective is {spec['story']}. Withdraw ${spec['withdrawal']:,.0f} from my existing portfolio and "
                f"rebalance the remaining assets to {t['STOCK']:.1%} stocks, {t['BOND']:.1%} bonds, and "
                f"{t['CASH']:.1%} cash. After withdrawal, keep at least ${spec['min_cash']:,.0f} in cash. "
                f"Do not sell {protected}. Trade whole shares of existing assets only. "
                "Do not borrow or add external deposits. Provide a valid trade plan.")
    (s_lo, s_hi), (b_lo, b_hi), (c_lo, c_hi) = spec["stock_range"], spec["bond_range"], spec["cash_range"]  # S03
    return ("Review my current portfolio and identify every investment policy violation. "
            f"Stocks must be between {s_lo:.0%} and {s_hi:.0%}; bonds between {b_lo:.0%} and {b_hi:.0%}; "
            f"cash between {c_lo:.0%} and {c_hi:.0%}. Cryptocurrency exposure is prohibited. "
            f"No individual stock may exceed {spec['individual_cap']:.0%} of the total portfolio. "
            "Report the violated rule, affected asset or class, observed percentage, and limit. "
            "If there are no violations, report none.")


def _set_portfolio(scenario, spec, keep_only_listed=False):
    portfolio = scenario["portfolio"]
    if keep_only_listed:
        portfolio["holdings"] = [h for h in portfolio["holdings"] if h["symbol"] in spec["quantities"]]
    elif {h["symbol"] for h in portfolio["holdings"]} != set(spec["quantities"]):
        raise ValueError(f"{spec['id']}: spec holdings differ from the base scenario")
    for h in portfolio["holdings"]:
        h["quantity"] = spec["quantities"][h["symbol"]]
        h["market_value"] = round(float(h["quantity"]) * float(h["current_price"]), 2)
    portfolio["cash_balance"] = float(spec["cash"])
    portfolio["total_value"] = round(sum(h["market_value"] for h in portfolio["holdings"]) + portfolio["cash_balance"], 2)


def _stamp(scenario, variant_id):
    """scenario_id is the variant's own name; there is no variant_id field."""
    scenario["scenario_id"] = variant_id
    scenario.pop("variant_id", None)
    scenario["output_requirements"]["response_schema"]["properties"]["scenario_id"]["enum"] = [variant_id]


def build_portfolio_variant(sid, spec):
    """S01, S02, S03, S05, S07, S08."""
    scenario = load_base(sid)
    _stamp(scenario, spec["id"])
    policy = scenario["investment_policy"]
    _set_portfolio(scenario, spec, keep_only_listed=(sid == "S03"))

    if sid == "S03":
        classes = [("STOCK", spec["stock_range"]), ("BOND", spec["bond_range"]), ("CASH", spec["cash_range"])]
        policy["restrictions"] = (
            [{"restriction_type": "CLASS_RANGE", "parameters_json": {"asset_class": c, "min_pct": lo, "max_pct": hi}}
             for c, (lo, hi) in classes]
            + [{"restriction_type": "PROHIBITED_CLASS", "parameters_json": {"asset_class": "CRYPTO", "max_pct": 0}},
               {"restriction_type": "MAX_INDIVIDUAL_STOCK", "parameters_json": {"max_pct": spec["individual_cap"]}}])
    else:
        managed = RULE_ORDER[sid]
        kept = [r for r in policy["restrictions"] if r["restriction_type"] not in managed]
        added = [(k, _managed_rule(k, spec)) for k in managed]
        policy["restrictions"] = kept + [{"restriction_type": k, "parameters_json": p} for k, p in added if p is not None]
        if "targets" in spec:
            policy["target_allocations"] = copy.deepcopy(spec["targets"])
    if sid in FLOW_FIELD:
        flow_type, field = FLOW_FIELD[sid]
        flows = scenario["cash_flows"]
        if len(flows) != 1 or flows[0]["flow_type"] != flow_type:
            raise ValueError(f"{spec['id']}: base scenario must hold exactly one {flow_type}")
        flows[0]["amount"] = float(spec[field])
    scenario["client_request"] = _request_text(sid, spec)
    return scenario


def build_fund_variant(spec):
    scenario = load_base("S04")
    _stamp(scenario, spec["id"])
    for fund, side in zip(scenario["investment_options"], "ab"):
        fund.update({"gross_annual_return": spec[f"{side}_return"], "annual_fee": spec[f"{side}_annual_fee"],
                     "entry_fee": spec[f"{side}_entry_fee"], "years": spec["years"]})
    scenario["client_request"] = (
        f"{spec['background']} Compare the {spec['years']}-year final values after all stated fees. "
        "Report each fund's final value, absolute difference, and better-performing fund.")
    scenario["story_type"] = spec["story_type"]
    return scenario


def build_retirement_variant(spec):
    scenario = load_base("S06")
    _stamp(scenario, spec["id"])
    months, rate = spec["total_months"], spec["monthly_rate"]
    scenario["portfolio"]["total_value"] = spec["initial_balance"]
    scenario["cash_flows"] = (
        [{"flow_type": "CONTRIBUTION", "amount": amount, "start_month": start, "duration_months": duration,
          "frequency": "MONTHLY", "execution_timing": "END_OF_MONTH"} for start, duration, amount in spec["contributions"]]
        + [{"flow_type": "WITHDRAWAL", "amount": amount, "start_month": month, "duration_months": 1,
            "frequency": "ONCE", "execution_timing": "END_OF_MONTH"} for month, amount in spec["withdrawals"]])
    scenario["retirement_assumptions"] = {
        "months": months, "nominal_annual_return": rate * 12, "compounds_per_year": 12, "monthly_interest_rate": rate,
        "target": spec["target_balance"], "additional_contribution_start_month": 1,
        "additional_contribution_duration_months": months, "additional_contribution_execution_timing": "END_OF_MONTH"}
    client = scenario["client"]
    client["investment_horizon_years"] = months / 12
    if spec["story_type"] == "Retirement planning":
        client["retirement_age"] = client["age"] + months / 12
    else:
        client.pop("retirement_age", None)
    scenario["client_request"] = (
        f"{spec['background']} Project the month-{months} balance, report the target shortfall, and calculate the "
        f"minimum fixed extra end-of-month contribution across months 1-{months}, rounded upward to cents.")
    scenario["story_type"] = spec["story_type"]
    return scenario


def build_scenario(sid, spec):
    if sid == "S04":
        return build_fund_variant(spec)
    if sid == "S06":
        return build_retirement_variant(spec)
    return build_portfolio_variant(sid, spec)


def build_variant(sid, spec):
    """Scenario JSON, per-variant ground truth and report for one spec row."""
    scenario = build_scenario(sid, spec)
    vid = spec["id"]
    head = {"scenario_id": vid}
    report = {**head, "verified": True, **{k: spec[k] for k in ("difficulty", "story", "story_type") if k in spec}}
    if sid == "S03":
        violations = policy_violations(scenario)
        if len(violations) != spec["expected_count"]:
            raise ValueError(f"{vid}: expected {spec['expected_count']} violations, found {len(violations)}")
        status = "VIOLATIONS_FOUND" if violations else "FEASIBLE"
        truth = {**head, "status": status, "trades": [], "violation_count": len(violations), "violations": violations}
        report.update(variant_type=spec["type"], violation_count=len(violations))
    elif sid in ("S04", "S06"):
        truth = {**head, **numeric_ground_truth(scenario)}
    else:
        solution = solve_trade_plan(scenario, whole_shares=True)
        if not solution["found"]:
            raise ValueError(f"{vid}: no feasible plan ({solution.get('message')})")
        v = solution["verification"]
        truth = {**head, "status": "FEASIBLE", "trades": solution["trades"], "verification": v}
        report.update(number_of_trades=len(solution["trades"]), final_allocations=v["allocations"], final_cash=v["final_cash"])
    return scenario, truth, report


def generate(out_dir, only=None):
    """Build the selected spec rows, then write three files per variant.

    Target files are checked first, and nothing is written unless every
    requested variant builds and verifies.
    """
    out_dir = Path(out_dir)
    selected = [(sid, spec) for sid, rows in load_specs().items() for spec in rows
                if only is None or spec["id"] in only or sid in only]
    ids = [spec["id"] for _, spec in selected]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("No variants selected, or duplicate variant ids in specs")
    targets = [out_dir / f"{vid}{suffix}.json" for vid in ids for suffix in ("", "_ground_truth", "_report")]
    blocked = [p.name for p in targets if is_approved(p) or p.exists()]
    if blocked:
        raise FileExistsError(f"Refusing to write over existing/approved files: {blocked[:5]}")
    built = []
    for sid, spec in selected:
        built.append((spec["id"], *build_variant(sid, spec)))
        print(f"BUILT | {spec['id']}")
    for vid, scenario, truth, report in built:
        write_json(out_dir / f"{vid}.json", scenario)
        write_json(out_dir / f"{vid}_ground_truth.json", truth)
        write_json(out_dir / f"{vid}_report.json", report)
    print(f"Wrote {len(built)} variants to {out_dir}")
    return ids


def compare(folder=DATA_DIR):
    """Read-only: rebuild each spec row and compare it with the scenario file in `folder`."""
    folder = Path(folder)
    same = different = 0
    for sid, rows in load_specs().items():
        for spec in rows:
            path = folder / f"{spec['id']}.json"
            if not path.exists():
                continue
            existing, rebuilt = read_json(path), build_scenario(sid, spec)
            changed = sorted(k for k in set(existing) | set(rebuilt) if existing.get(k) != rebuilt.get(k))
            same, different = same + (not changed), different + bool(changed)
            print(f"{'DIFF' if changed else 'SAME'} | {spec['id']}" + (f" | fields: {changed}" if changed else ""))
    print(f"\nIdentical: {same} | different: {different}")
    return different == 0


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------
REQUIRED_FIELDS = {"scenario_id", "schema_version", "answer_type", "portfolio", "investment_policy",
                   "cash_flows", "client_request", "assumptions", "output_requirements"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _close(a, b, tol=0.015):
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol)


def _check_response_schema(schema, errors, seen):
    try:
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(schema)
    except ImportError:
        seen["no_jsonschema"] = True
    except Exception as exc:
        errors.append(f"Invalid response schema: {exc}")


def audit_variant(variant_id, folder, seen):
    """Return (errors, warnings) for one variant; `seen` collects cross-variant state."""
    errors, warnings = [], []
    sid = "S" + variant_id[1:3]
    scenario = read_json(folder / f"{variant_id}.json")
    truth_path = folder / f"{variant_id}_ground_truth.json"
    if not truth_path.exists():
        return [f"Missing {truth_path.name}"], warnings
    truth = read_json(truth_path)
    truth = truth.get("ground_truth", truth)
    if not (folder / f"{variant_id}_report.json").exists():
        errors.append("Missing report file")

    missing = REQUIRED_FIELDS - set(scenario)
    if missing:
        return [f"Missing fields: {sorted(missing)}"], warnings
    expected_id = f"scenario_id must be '{variant_id}' and there must be no variant_id"
    if scenario["scenario_id"] != variant_id or "variant_id" in scenario:
        errors.append(f"scenario file: {expected_id} (found scenario_id={scenario['scenario_id']!r}, variant_id={scenario.get('variant_id')!r})")
    if truth.get("scenario_id") != variant_id or "variant_id" in truth:
        errors.append(f"ground-truth file: {expected_id} (found scenario_id={truth.get('scenario_id')!r}, variant_id={truth.get('variant_id')!r})")
    if scenario["answer_type"] != EXPECTED_ANSWER_TYPE[sid]:
        errors.append(f"answer_type {scenario['answer_type']} != {EXPECTED_ANSWER_TYPE[sid]}")
    contract = scenario["output_requirements"]
    if contract.get("contract_id") != CONTRACT_ID:
        errors.append("Wrong contract_id")
    enum_field = contract["response_schema"]["properties"]["scenario_id"]
    if enum_field.get("enum") != [variant_id]:
        errors.append(f"response_schema scenario_id enum {enum_field.get('enum')} != ['{variant_id}']")
    seen["contracts"].add(_canonical({**contract, "response_schema": {**contract["response_schema"], "properties": {
        **contract["response_schema"]["properties"], "scenario_id": {**enum_field, "enum": None}}}}))  # shared part only
    seen["schema_versions"].add(scenario["schema_version"])
    _check_response_schema(contract["response_schema"], errors, seen)

    portfolio = scenario["portfolio"]
    if portfolio["holdings"]:
        for h in portfolio["holdings"]:
            if abs(h["quantity"] * h["current_price"] - h["market_value"]) > CASH_TOL:
                errors.append(f"market_value mismatch: {h['symbol']}")
            if h["quantity"] < 0 or h["current_price"] <= 0:
                errors.append(f"Invalid quantity/price: {h['symbol']}")
        total = holdings_value(portfolio["holdings"]) + portfolio["cash_balance"]
        if abs(total - portfolio["total_value"]) > TOTAL_TOL:
            errors.append(f"total_value {portfolio['total_value']} != computed {total:.2f}")

    signature = _canonical({k: v for k, v in scenario.items() if k not in ("scenario_id", "variant_id", "output_requirements")})
    if signature in seen["signatures"]:
        errors.append(f"Duplicate of {seen['signatures'][signature]}")
    seen["signatures"][signature] = variant_id

    try:
        if sid == "S03":
            expected = policy_violations(scenario)
            got = {(v["kind"], v["scope"]): v for v in truth.get("violations", [])}
            if set(got) != {(v["kind"], v["scope"]) for v in expected}:
                errors.append("Ground-truth violations differ from recomputed violations")
            for v in expected:
                g = got.get((v["kind"], v["scope"]))
                if g and not (_close(g["observed"], v["observed"], 5e-4) and _close(g["limit"], v["limit"], 1e-9)):
                    errors.append(f"Violation numbers differ: {v['kind']} {v['scope']}")
        elif sid == "S04":
            r = fund_comparison(scenario)
            for name, value in r["values"].items():
                if not _close(truth["fund_values"][name], value):
                    errors.append(f"Fund value differs: {name}")
            if truth["winner"] != r["winner"] or not _close(truth["absolute_difference"], r["difference"]):
                errors.append("Winner or difference differs")
        elif sid == "S06":
            r = retirement_truth(scenario)
            if not _close(truth["projected_balance"], r["projected_balance"]):
                errors.append("Projected balance differs")
            if not _close(truth["extra_monthly_rounded_up"], r["extra_monthly_rounded_up"], 1e-9):
                errors.append("Extra monthly contribution differs")
            if scenario.get("story_type") != "Retirement planning" and (
                    "retire" in scenario["client_request"].lower() or "retirement_age" in scenario["client"]):
                warnings.append("Retirement wording in a non-retirement variant")
        else:
            replay = verify_trade_plan(scenario, truth["trades"])
            if not replay["valid"]:
                errors += [f"Ground-truth plan invalid: {e}" for e in replay["errors"]]
            if not solve_trade_plan(scenario, time_limit=30, whole_shares=True)["found"]:
                errors.append("Independent solver finds no feasible plan")
    except Exception as exc:  # report, never crash the whole audit
        errors.append(f"{type(exc).__name__}: {exc}")
    return errors, warnings


def audit(folder=DATA_DIR):
    """Audit every T0X-v### scenario file in `folder`. Returns True when all pass."""
    folder = Path(folder)
    ids = sorted(p.stem for p in folder.glob("T??-v???.json") if VARIANT_RE.fullmatch(p.stem))
    seen = {"contracts": set(), "schema_versions": set(), "signatures": {}}
    failed = warned = 0
    for vid in ids:
        errors, warnings = audit_variant(vid, folder, seen)
        print(f"{'FAIL' if errors else 'PASS'} | {vid}")
        for message in errors:
            print(f"    ERROR: {message}")
        for message in warnings:
            print(f"    WARNING: {message}")
        failed += bool(errors)
        warned += len(warnings)
    global_errors = []
    if len(seen["contracts"]) > 1:
        global_errors.append("output_requirements differ between variants")
    if len(seen["schema_versions"]) > 1:
        global_errors.append(f"mixed schema_version values: {sorted(seen['schema_versions'])}")
    if seen.get("no_jsonschema"):
        print("WARNING: jsonschema not installed; response schema syntax was not validated")
        warned += 1
    extra = sorted(p.name for p in folder.glob("*.json") if not re.fullmatch(r"T0[1-8]-v\d{3}(_ground_truth|_report)?\.json", p.name))
    if extra:
        print(f"WARNING: unexpected JSON files: {extra}")
        warned += 1
    for message in global_errors:
        print(f"ERROR: {message}")
    ok = not failed and not global_errors and bool(ids)
    print(f"\nVariants checked: {len(ids)} | passed: {len(ids) - failed} | failed: {failed} | warnings: {warned}")
    print(f"AUDIT: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------
# Response-contract test
# --------------------------------------------------------------------------
def _observation(subject, measure, value, rounding="NONE"):
    return {"subject": subject, "measure": measure, "stage": "FINAL", "rounding": rounding,
            "unit": "MONEY", "value": float(value)}


def reference_response(scenario, truth):
    """Build the response a perfect agent would return, from the ground truth."""
    response = {"scenario_id": scenario["scenario_id"], "answer_type": scenario["answer_type"], "status": "FEASIBLE",
                "trades": [], "metrics": {"observations": [], "selection": None}, "violations": [],
                "explanation": "Result calculated from the supplied scenario inputs."}
    obs = response["metrics"]["observations"]
    if scenario["answer_type"] == "TRADE_PLAN":
        response["trades"] = truth["trades"]
    elif scenario["answer_type"] == "VIOLATION_LIST":
        response["violations"] = truth["violations"]
        response["status"] = "VIOLATIONS_FOUND" if truth["violations"] else "FEASIBLE"
    elif scenario.get("calculation_type") == "FUND_COMPARISON":
        obs += [_observation(n, "VALUE", v) for n, v in truth["fund_values"].items()]
        obs.append(_observation("OPTIONS", "DIFFERENCE", truth["absolute_difference"]))
        response["metrics"]["selection"] = truth["winner"]
    else:
        obs += [_observation("PORTFOLIO", "VALUE", truth["projected_balance"]),
                _observation("PORTFOLIO", "DIFFERENCE", truth["shortfall"]),
                _observation("PORTFOLIO", "PAYMENT", truth["extra_monthly_exact"]),
                _observation("PORTFOLIO", "PAYMENT", truth["extra_monthly_rounded_up"], "UPWARD_CENT")]
    return response


def contract_test(folder=DATA_DIR):
    """Every variant's reference response must pass its own schema; three
    deliberately broken responses per variant must be rejected."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        print("ERROR: jsonschema is not installed (python3 -m pip install jsonschema)")
        return False
    folder = Path(folder)
    ids = sorted(p.stem for p in folder.glob("T??-v???.json") if VARIANT_RE.fullmatch(p.stem))
    passed = 0
    for vid in ids:
        try:
            scenario = read_json(folder / f"{vid}.json")
            truth = read_json(folder / f"{vid}_ground_truth.json")
            truth = truth.get("ground_truth", truth)
            validator = Draft202012Validator(scenario["output_requirements"]["response_schema"])
            response = reference_response(scenario, truth)
            validator.validate(response)
            broken = {"status": {**response, "status": "UNKNOWN_STATUS"},
                      "metrics": {k: v for k, v in response.items() if k != "metrics"},
                      "extra field": {**response, "secret_answer": 123}}
            for name, bad in broken.items():
                if validator.is_valid(bad):
                    raise AssertionError(f"schema accepted a response with bad {name}")
            passed += 1
            print(f"PASS | {vid}")
        except Exception as exc:
            print(f"FAIL | {vid} | {type(exc).__name__}: {exc}")
    print(f"\nResponse contract: {passed}/{len(ids)}")
    return passed == len(ids) and bool(ids)


# --------------------------------------------------------------------------
# Consolidated evaluation files: ground_truth.json, checks.json, metric_mapping.json
# Keyed by variant_id. Existing files are extended, never silently rewritten.
# --------------------------------------------------------------------------
GROUND_TRUTH_NAME, CHECKS_NAME, MAPPING_NAME = "ground_truth.json", "checks.json", "metric_mapping.json"
REFERENCE_NOTE = "Reference example; any plan satisfying all requirements is acceptable."
ENTRY_FIELDS = ("scenario_id", "answer_type", "status", "trades", "metrics", "violations", "explanation")


def _num(x):
    x = round(float(x), 6)
    return int(x) if x.is_integer() else x


def _pct(x):
    return f"{x * 100:.2f}%".replace(".00%", "%")


def _trade_metrics(scenario, truth):
    replay = verify_trade_plan(scenario, truth["trades"])
    if not replay["valid"]:
        raise ValueError(f"reference plan is invalid: {replay['errors']}")
    sid = base_scenario_id(identity(scenario))
    rules = rules_of(scenario["investment_policy"])
    held = {h["symbol"]: h for h in replay["final_holdings"]}
    cash0 = float(scenario["portfolio"]["cash_balance"])
    cash = replay["final_cash"]
    total = sum(h["quantity"] * h["current_price"] for h in held.values()) + cash
    by_class = {"CASH": cash}
    for h in held.values():
        by_class[h["asset_class"]] = by_class.get(h["asset_class"], 0) + h["quantity"] * h["current_price"]

    metrics = {"initial_total": _num(scenario["portfolio"]["total_value"]), "final_total": _num(total)}
    if sid in ("S02", "S08"):
        metrics["cash_before_withdrawal"] = _num(replay["cash_before_withdrawal"])
    else:
        metrics["cash_before_trades"] = _num(cash0 + replay["deposits"])
    metrics["final_cash"] = _num(cash)
    metrics["allocations"] = {c: v / total for c, v in by_class.items()}
    metrics["final_values"] = {s: _num(h["quantity"] * h["current_price"]) for s, h in held.items()}
    metrics["final_quantities"] = {s: _num(h["quantity"]) for s, h in held.items()}
    if sid == "S02":
        reserve = max(p["amount"] for p in rules["MIN_CASH"])
        metrics["additional_liquidity_needed"] = _num(max(0, replay["withdrawals"] + reserve - cash0))
    if sid == "S05":
        tech = [s for p in rules["MAX_DIRECT_SECTOR"] for s in p["symbols"]]
        metrics["direct_technology_pct"] = sum(held[s]["quantity"] * held[s]["current_price"] for s in tech) / total
    if sid == "S07":
        metrics["historical_total"] = _num(cash0 + sum(h["quantity"] * h["historical_price"]
                                                       for h in scenario["portfolio"]["holdings"]))
    return metrics, [], REFERENCE_NOTE


def _violation_metrics(scenario, truth):
    portfolio = scenario["portfolio"]
    total = portfolio["total_value"]
    values = class_values(portfolio["holdings"], portfolio["cash_balance"])
    mine = {(v["kind"], v["scope"]): v for v in policy_violations(scenario)}
    given = {(v["kind"], v["scope"]): v for v in truth["violations"]}
    if set(mine) != set(given):
        raise ValueError(f"violations differ from recomputed ones: {sorted(set(mine) ^ set(given))}")
    for key, v in mine.items():
        if abs(v["observed"] - given[key]["observed"]) > 5e-4 or abs(v["limit"] - given[key]["limit"]) > EPS:
            raise ValueError(f"violation numbers differ: {key}")
    metrics = {"initial_total": _num(total), "allocations": {c: v / total for c, v in values.items()},
               "violation_count": len(truth["violations"])}
    wording = {"CLASS_MAX": "overweight", "CLASS_MIN": "underweight", "INDIVIDUAL_MAX": "individual concentration",
               "PROHIBITED_CLASS": "prohibited"}
    if truth["violations"]:
        parts = [f"{v['scope']} {wording[v['kind']]} ({_pct(v['observed'])} vs limit {_pct(v['limit'])})"
                 for v in truth["violations"]]
        text = f"Exactly {len(parts)} violation{'s' if len(parts) != 1 else ''}: " + "; ".join(parts) + ". No remediation trades."
    else:
        text = "No policy violations: every asset class and individual stock is within its limit. Report an empty violation list and no trades."
    return metrics, truth["violations"], text


def _fund_metrics(scenario, truth):
    r = fund_comparison(scenario)
    for name, value in r["values"].items():
        if abs(value - truth["fund_values"][name]) > 0.01:
            raise ValueError(f"{name} differs from the ground-truth file")
    if r["winner"] != truth["winner"] or abs(r["difference"] - truth["absolute_difference"]) > 0.01:
        raise ValueError("winner or difference differs from the ground-truth file")
    a, b = scenario["investment_options"]
    formula = lambda o: (f"{o['initial_investment'] * (1 - o['entry_fee']):g}*((1+{o['gross_annual_return']:g})"
                         f"*(1-{o['annual_fee']:g}))^{o['years']}")
    metrics = {"initial_total": _num(scenario["portfolio"]["total_value"]), "fund_a_final": r["values"]["Fund A"],
               "fund_b_final": r["values"]["Fund B"], "absolute_difference": r["difference"], "better_fund": r["winner"]}
    return metrics, [], ("Entry fee deducted first. Annual gross return and annual fee applied multiplicatively: "
                         f"A = {formula(a)}; B = {formula(b)}.")


def _retirement_metrics(scenario, truth):
    r = retirement_truth(scenario)
    if abs(r["projected_balance"] - truth["projected_balance"]) > 0.01 or r["extra_monthly_rounded_up"] != truth["extra_monthly_rounded_up"]:
        raise ValueError("projection differs from the ground-truth file")
    months = scenario["retirement_assumptions"]["months"]
    metrics = {"initial_total": _num(scenario["portfolio"]["total_value"]), "final_balance": r["projected_balance"],
               "target_shortfall": r["target_minus_projected"], "additional_monthly_contribution": r["extra_monthly_exact"],
               "additional_monthly_contribution_cents": r["extra_monthly_rounded_up"]}
    return metrics, [], ("Monthly growth precedes end-of-month flows; scheduled withdrawals and contributions follow growth. "
                         f"Extra end-of-month payment applies for months 1-{months}. "
                         "Target shortfall means target minus projected balance.")


def ground_truth_entry(scenario, truth):
    """Consolidated ground-truth entry for one variant (metrics derived from its scenario)."""
    vid = identity(scenario)
    sid = base_scenario_id(vid)
    if identity(truth) != vid:
        raise ValueError(f"ground-truth file id {identity(truth)!r} does not match the scenario {vid!r}")
    build = {"S03": _violation_metrics, "S04": _fund_metrics, "S06": _retirement_metrics}.get(sid, _trade_metrics)
    metrics, violations, text = build(scenario, truth)
    return {"scenario_id": vid, "answer_type": scenario["answer_type"], "status": truth["status"],
            "trades": truth["trades"], "metrics": metrics, "violations": violations, "explanation": text}


def checks_for(variant_id, template, scenario, entry):
    """Checks for one variant: the base scenario's checks with three values taken from the variant itself.

    STATUS follows the ground truth, crypto_count counts CRYPTO violations in the ground truth, and the
    replay's target_allocation_tolerance is the scenario's allocation_tolerance (null without targets).
    """
    base = base_scenario_id(variant_id)
    checks = copy.deepcopy(template)
    policy = scenario["investment_policy"]
    for check in checks:
        check["check_id"] = variant_id + check["check_id"][len(base):]
        params = check["params_json"]
        if check["check_type"] == "STATUS":
            params["expected"] = entry["status"]
        elif check["check_type"] == "EXACT_SCOPED_VIOLATIONS":
            params["crypto_count"] = sum(v["scope"] == "CRYPTO" for v in entry["violations"])
        elif check["check_type"] == "REPLAY_TRADES":
            params["target_allocation_tolerance"] = policy.get("allocation_tolerance") if policy.get("target_allocations") else None
    return checks


def _merge(existing, incoming, replace):
    """Add new keys; identical keys are skipped; differing keys are conflicts unless replace=True."""
    merged = dict(existing)
    result = {"added": [], "same": [], "replaced": [], "conflicts": []}
    for key, value in incoming.items():
        if key not in merged:
            merged[key] = value
            result["added"].append(key)
        elif merged[key] == value:
            result["same"].append(key)
        elif replace:
            merged[key] = value
            result["replaced"].append(key)
        else:
            result["conflicts"].append(key)
    return merged, result


def _ordered(data):
    """Scenario ids (S01...) first, then variant ids (T01-v001...); the order is only cosmetic."""
    return {k: data[k] for k in sorted(data, key=lambda key: (key[0] != "S", key))}


def _write_with_backup(path, data):
    if path.exists():
        backup = path.with_name(path.name + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)  # the first backup is kept, never overwritten
    write_json(path, data, overwrite=True)


def consolidate(folder=DATA_DIR, out_dir=CONSOLIDATED_DIR, checks_template=None, mapping_template=None, replace=False):
    """Create or extend ground_truth.json, checks.json and metric_mapping.json in `out_dir`.

    Reads <id>.json and <id>_ground_truth.json for every variant in `folder`. Nothing is written
    unless every variant converts and no existing entry would change (use replace=True to allow that).
    """
    folder, out_dir = Path(folder), Path(out_dir)
    paths = {name: out_dir / name for name in (GROUND_TRUTH_NAME, CHECKS_NAME, MAPPING_NAME)}

    def template(given, name):
        """Base-scenario entries: the given file, else the out-dir file and ./<name> (first one wins)."""
        sources = [Path(given)] if given else [paths[name], ROOT / name]
        merged = {}
        for path in sources:
            if path.exists():
                data = read_json(path)
                for key, value in (data.get("scenarios", {}) if name == MAPPING_NAME else data).items():
                    merged.setdefault(key, value)
        head = next((read_json(p) for p in sources if p.exists()), {}) if name == MAPPING_NAME else {}
        return merged, {k: v for k, v in head.items() if k != "scenarios"}

    base_checks, _ = template(checks_template, CHECKS_NAME)
    base_routes, mapping_head_default = template(mapping_template, MAPPING_NAME)
    ids = sorted(p.stem for p in folder.glob("T??-v???.json") if VARIANT_RE.fullmatch(p.stem))
    if not ids:
        print(f"No variant files found in {folder}")
        return False

    gt, checks, routes, errors = {}, {}, {}, []
    for vid in ids:
        try:
            scenario = read_json(folder / f"{vid}.json")
            truth = read_json(folder / f"{vid}_ground_truth.json")
            if identity(scenario) != vid:
                raise ValueError(f"scenario file id {identity(scenario)!r} does not match its file name")
            sid = base_scenario_id(vid)
            if sid not in base_checks or sid not in base_routes:
                raise KeyError(f"no checks/mapping template for {sid}")
            gt[vid] = ground_truth_entry(scenario, truth)
            checks[vid] = checks_for(vid, base_checks[sid], scenario, gt[vid])
            routes[vid] = copy.deepcopy(base_routes[sid])  # full copy: no inheritance assumed
        except Exception as exc:
            errors.append(f"{vid}: {type(exc).__name__}: {exc}")
    if errors:
        print("\n".join(f"ERROR | {e}" for e in errors))
        return False

    existing = {name: read_json(p) if p.exists() else None for name, p in paths.items()}
    mapping_head = {k: v for k, v in (existing[MAPPING_NAME] or mapping_head_default).items() if k != "scenarios"}
    merged, results = {}, {}
    merged[GROUND_TRUTH_NAME], results[GROUND_TRUTH_NAME] = _merge(existing[GROUND_TRUTH_NAME] or {}, gt, replace)
    merged[CHECKS_NAME], results[CHECKS_NAME] = _merge(existing[CHECKS_NAME] or {}, checks, replace)
    scenarios, results[MAPPING_NAME] = _merge((existing[MAPPING_NAME] or {}).get("scenarios", {}), routes, replace)
    merged[MAPPING_NAME] = {**mapping_head, "scenarios": _ordered(scenarios)}
    merged[GROUND_TRUTH_NAME], merged[CHECKS_NAME] = _ordered(merged[GROUND_TRUTH_NAME]), _ordered(merged[CHECKS_NAME])

    conflicts = {name: r["conflicts"] for name, r in results.items() if r["conflicts"]}
    if conflicts:
        for name, keys in conflicts.items():
            print(f"CONFLICT | {name}: existing entries differ from the new ones: {keys}")
        print("Nothing written. Re-run with --replace to overwrite those entries.")
        return False
    for name, path in paths.items():
        r = results[name]
        if r["added"] or r["replaced"] or existing[name] is None:
            _write_with_backup(path, merged[name])
        print(f"{name}: added {len(r['added'])}, unchanged {len(r['same'])}, replaced {len(r['replaced'])}")

    print("\nCOMPATIBILITY (the scorer must be tested on these, not assumed):")
    for vid, entry in gt.items():
        if entry["answer_type"] == "VIOLATION_LIST":
            zero = sum(v["scope"] == "CRYPTO" for v in entry["violations"]) == 0
            print(f"  {vid}: status {entry['status']}, {len(entry['violations'])} violations"
                  + (", crypto_count 0" if zero else ""))
    return check_consistency(out_dir, folder)


def check_consistency(out_dir=CONSOLIDATED_DIR, folder=None):
    """The three files must agree: every ground-truth entry has checks and a mapping, and each
    required metric exists in the ground truth. `folder` (optional) enables scenario-level checks."""
    out_dir = Path(out_dir)
    try:
        gt, checks, mapping = (read_json(out_dir / n) for n in (GROUND_TRUTH_NAME, CHECKS_NAME, MAPPING_NAME))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return False
    routes = mapping.get("scenarios", {})
    errors, warnings = [], []
    for name, keys in (("checks", checks), ("metric_mapping", routes)):
        warnings += [f"{name} has '{k}' but ground_truth does not" for k in keys if k not in gt]

    for key, entry in gt.items():
        errors += [f"{key}: ground truth lacks '{f}'" for f in ENTRY_FIELDS if f not in entry]
        if entry.get("scenario_id") != key or "variant_id" in entry:
            errors.append(f"{key}: scenario_id must equal the key and there must be no variant_id "
                          f"(found scenario_id={entry.get('scenario_id')!r}, variant_id={entry.get('variant_id')!r})")
        if key not in checks:
            errors.append(f"{key}: no checks entry")
        if key not in routes:
            errors.append(f"{key}: no metric_mapping entry")
        metrics = entry.get("metrics", {})
        scenario_path = Path(folder) / f"{key}.json" if folder else None
        scenario = read_json(scenario_path) if scenario_path and scenario_path.exists() else None
        for check in checks.get(key, []):
            kind, params = check["check_type"], check["params_json"]
            if not check["check_id"].startswith(key + "_"):
                errors.append(f"{key}: check_id '{check['check_id']}' does not start with the key")
            if kind == "REQUIRED_METRICS":
                errors += [f"{key}: required metric '{m}' missing from ground truth" for m in params["keys"] if m not in metrics]
                needed = routes.get(key, {}).get("required_private_metrics")
                if needed is not None and set(needed) != set(params["keys"]):
                    errors.append(f"{key}: mapping required_private_metrics differ from checks keys")
                errors += [f"{key}: mapping has no route for '{m}'" for m in params["keys"]
                           if m not in routes.get(key, {}).get("metric_routes", {})]
            elif kind == "STATUS" and params["expected"] != entry.get("status"):
                errors.append(f"{key}: STATUS expects {params['expected']} but ground truth is {entry.get('status')}")
            elif kind == "EMPTY_TRADES" and params["expected"] != entry.get("trades"):
                errors.append(f"{key}: EMPTY_TRADES does not match ground-truth trades")
            elif kind == "EXACT_SCOPED_VIOLATIONS":
                actual = sum(v["scope"] == "CRYPTO" for v in entry.get("violations", []))
                if params.get("crypto_count") != actual:
                    errors.append(f"{key}: crypto_count {params.get('crypto_count')} != {actual} in ground truth")
            elif kind == "REPLAY_TRADES" and scenario is not None:
                policy = scenario["investment_policy"]
                expected = policy.get("allocation_tolerance") if policy.get("target_allocations") else None
                if params["target_allocation_tolerance"] != expected:
                    errors.append(f"{key}: target_allocation_tolerance {params['target_allocation_tolerance']} != scenario {expected}")
    for message in errors:
        print(f"ERROR: {message}")
    for message in warnings:
        print(f"WARNING: {message}")
    print(f"\nCONSISTENCY: {'PASS' if not errors else 'FAIL'} | ground-truth entries: {len(gt)} | errors: {len(errors)} | warnings: {len(warnings)}")
    return not errors


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "contract-test", "compare"):
        sub.add_parser(name).add_argument("--dir", default=str(DATA_DIR))
    gen = sub.add_parser("generate", help="build variants from variant_specs.json into a NEW folder")
    gen.add_argument("--out", default=str(STAGING_DIR))
    gen.add_argument("--only", nargs="*", help="variant ids or scenario ids (e.g. T01-v005 S03)")
    con = sub.add_parser("consolidate", help="create or extend ground_truth.json, checks.json, metric_mapping.json")
    con.add_argument("--dir", default=str(DATA_DIR), help="folder with the variant files")
    con.add_argument("--out-dir", default=str(CONSOLIDATED_DIR), help="where the three files live (extended if present)")
    con.add_argument("--checks-template", help="checks file holding S01-S08 (default: the out-dir file, else ./checks.json)")
    con.add_argument("--mapping-template", help="mapping file holding S01-S08 (default: the out-dir file, else ./metric_mapping.json)")
    con.add_argument("--replace", action="store_true", help="overwrite existing entries that differ")
    cc = sub.add_parser("check-consistency", help="verify the three consolidated files agree")
    cc.add_argument("--out-dir", default=str(CONSOLIDATED_DIR))
    cc.add_argument("--dir", help="optional variant folder for scenario-level checks")
    sub.add_parser("manifest-create")
    sub.add_parser("manifest-check")
    args = parser.parse_args(argv)

    if args.command == "audit":
        return 0 if audit(args.dir) else 1
    if args.command == "contract-test":
        return 0 if contract_test(args.dir) else 1
    if args.command == "compare":
        return 0 if compare(args.dir) else 1
    if args.command == "generate":
        generate(args.out, set(args.only) if args.only else None)
        return 0
    if args.command == "consolidate":
        return 0 if consolidate(args.dir, args.out_dir, args.checks_template, args.mapping_template, args.replace) else 1
    if args.command == "check-consistency":
        return 0 if check_consistency(args.out_dir, args.dir) else 1
    if args.command == "manifest-create":
        manifest_create()
        return 0
    return 0 if manifest_check() else 1


if __name__ == "__main__":
    sys.exit(main())
