
import json
import copy
import random
from pathlib import Path

# Configuration
SCENARIO_ID = "S01"
BASE_DIR = Path(".")
OUTPUT_DIR = Path("generated_variants")


# Load a scenario template from the supported local paths.
def load_scenario(scenario_id):
    possible_paths = [
        BASE_DIR / "data" / "scenarios" / f"{scenario_id}.json",
        BASE_DIR / f"{scenario_id}.json"
    ]

    for filepath in possible_paths:
        if filepath.is_file():
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)

    raise FileNotFoundError(
        f"Scenario {scenario_id} not found in: {possible_paths}"
    )


# Update holding market values and the portfolio total in place.
def recalculate_portfolio(portfolio):
    total_holdings = 0

    for asset in portfolio["holdings"]:
        asset["market_value"] = (
            asset["quantity"] * asset["current_price"]
        )

        total_holdings += asset["market_value"]

    portfolio["total_value"] = (
        total_holdings + portfolio["cash_balance"]
    )


# Create an independent copy of a scenario before modifying it.
def create_variant(scenario):
    variant = copy.deepcopy(scenario)
    return variant

# Save a JSON variant without overwriting an existing file.
def save_variant(variant, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filepath = OUTPUT_DIR / filename

    if filepath.exists():
        raise FileExistsError(
            f"File already exists: {filepath}"
        )

    with open(filepath, "x", encoding="utf-8") as file:
        json.dump(variant, file, indent=2)

    print("Saved:", filepath)


# Read the individual-stock concentration limit from policy.
def get_max_individual_stock(scenario):
    restrictions = scenario["investment_policy"]["restrictions"]

    for restriction in restrictions:
        if restriction["restriction_type"] == "MAX_INDIVIDUAL_STOCK":
            return restriction["parameters_json"]["max_pct"]

    raise ValueError("MAX_INDIVIDUAL_STOCK restriction not found")



# Build a variant at the individual-stock limit.
def generate_boundary_variant(scenario, symbol="AAPL"):
    max_pct = get_max_individual_stock(scenario)

    portfolio = scenario["portfolio"]
    total_value = portfolio["total_value"]

    asset = next(
        (a for a in portfolio["holdings"] if a["symbol"] == symbol),
        None
    )

    if asset is None:
        raise ValueError(f"Asset {symbol} not found")

    if asset["asset_type"] != "INDIVIDUAL_STOCK":
        raise ValueError("Asset must be an individual stock")

    target_value = total_value * max_pct
    target_quantity = target_value / asset["current_price"]

    if not target_quantity.is_integer():
        raise ValueError(
            "Exact boundary cannot be reached with whole shares"
        )

    variant = set_asset_quantity(
        scenario, symbol, int(target_quantity)
    )

    actual_pct = (
        next(a["market_value"]
             for a in variant["portfolio"]["holdings"]
             if a["symbol"] == symbol)
        / variant["portfolio"]["total_value"]
    )

    assert abs(actual_pct - max_pct) < 1e-9

    print(f"{symbol} boundary: {actual_pct:.2%}")

    return variant



# Create the existing S01-specific boundary test case.
def generate_s01_boundary(scenario):
    variant = create_variant(scenario)

    holdings = {
        asset["symbol"]: asset
        for asset in variant["portfolio"]["holdings"]
    }

    holdings["TSLA"]["quantity"] = 40
    holdings["VOO"]["quantity"] = 36

    recalculate_portfolio(variant["portfolio"])

    original_total = scenario["portfolio"]["total_value"]
    new_total = variant["portfolio"]["total_value"]

    if new_total != original_total:
        raise ValueError("Portfolio total changed unexpectedly")

    if variant == scenario:
        raise ValueError("Variant is identical to original")

    return variant



# Check quantities, holding values, cash, and total consistency.
def validate_portfolio(portfolio):
    holdings = portfolio["holdings"]

    if portfolio["cash_balance"] < 0:
        raise ValueError("Cash balance cannot be negative")

    calculated_total = portfolio["cash_balance"]

    for asset in holdings:
        quantity = asset["quantity"]
        price = asset["current_price"]
        market_value = asset["market_value"]

        if (
            type(quantity) not in (int, float)
            or quantity < 0
            or (
                asset["asset_type"] == "INDIVIDUAL_STOCK"
                and not float(quantity).is_integer()
            )
        ):
            raise ValueError(
                f"Invalid quantity for {asset['symbol']}"
            )

        if price < 0:
            raise ValueError(
                f"Negative price for {asset['symbol']}"
            )

        expected_value = quantity * price

        if abs(market_value - expected_value) > 0.01:
            raise ValueError(
                f"Incorrect market value for {asset['symbol']}"
            )

        calculated_total += market_value

    if abs(calculated_total - portfolio["total_value"]) > 0.01:
        raise ValueError("Portfolio total is incorrect")

    print("Portfolio validation passed!")
    return True


# Index investment restrictions by their restriction type.
def get_portfolio_restrictions(scenario):
    restrictions = scenario["investment_policy"]["restrictions"]

    result = {}

    for restriction in restrictions:
        restriction_type = restriction["restriction_type"]
        parameters = restriction.get("parameters_json", {})

        result[restriction_type] = parameters

    return result



# Find holdings above the individual-stock limit.
def check_individual_stock_limits(scenario):
    restrictions = get_portfolio_restrictions(scenario)

    max_pct = restrictions["MAX_INDIVIDUAL_STOCK"]["max_pct"]

    portfolio = scenario["portfolio"]
    total_value = portfolio["total_value"]

    if total_value <= 0:
        raise ValueError("Portfolio total must be positive")

    violations = []

    for asset in portfolio["holdings"]:
        if asset["asset_type"] != "INDIVIDUAL_STOCK":
            continue

        actual_pct = asset["market_value"] / total_value

        if actual_pct > max_pct + 1e-9:        
            violations.append({
            "restriction_type": "MAX_INDIVIDUAL_STOCK",
            "symbol": asset["symbol"],
            "actual_pct": round(actual_pct, 4),
            "max_pct": max_pct
            })


    return violations


# Change one holding while adjusting cash to preserve value.
def set_asset_quantity(scenario, symbol, new_quantity):
    variant = create_variant(scenario)
    portfolio = variant["portfolio"]

    if type(new_quantity) is not int or new_quantity < 0:
        raise ValueError("Quantity must be a non-negative integer")

    asset = next(
        (a for a in portfolio["holdings"] if a["symbol"] == symbol),
        None
    )

    if asset is None:
        raise ValueError(f"Asset {symbol} not found")

    old_value = asset["market_value"]
    new_value = new_quantity * asset["current_price"]

    portfolio["cash_balance"] -= new_value - old_value

    if portfolio["cash_balance"] < 0:
        raise ValueError("Not enough cash for this change")

    asset["quantity"] = new_quantity

    recalculate_portfolio(portfolio)
    validate_portfolio(portfolio)

    return variant


# Generate a below, at, or above-limit stock test case.
def generate_stock_limit_variant(scenario, symbol, strategy):
    if strategy not in ("below", "boundary", "above"):
        raise ValueError("Invalid strategy")

    portfolio = scenario["portfolio"]
    max_pct = get_max_individual_stock(scenario)

    asset = next(
        (a for a in portfolio["holdings"] if a["symbol"] == symbol),
        None
    )

    if asset is None:
        raise ValueError(f"Asset {symbol} not found")

    if asset["asset_type"] != "INDIVIDUAL_STOCK":
        raise ValueError("Asset must be an individual stock")

    total = portfolio["total_value"]
    price = asset["current_price"]

    if total <= 0 or price <= 0:
        raise ValueError("Total and price must be positive")

    target_shares = total * max_pct / price

    if strategy == "below":
        quantity = max(0, int(target_shares - 1e-9))
    elif strategy == "above":
        quantity = int(target_shares + 1e-9) + 1
    else:
        if not target_shares.is_integer():
            raise ValueError("Exact boundary is not reachable")
        quantity = int(target_shares)

    variant = set_asset_quantity(scenario, symbol, quantity)

    actual_pct = (
        quantity * price / variant["portfolio"]["total_value"]
    )

    if strategy == "below":
        assert actual_pct < max_pct
    elif strategy == "boundary":
        assert abs(actual_pct - max_pct) < 1e-9
    else:
        assert actual_pct > max_pct

    return variant



# Compare holdings to detect sales of protected assets.
def check_no_sell_violations(original, variant):
    restrictions = get_portfolio_restrictions(original)
    protected_symbols = restrictions.get("NO_SELL", {}).get("symbols", [])

    original_holdings = {
        asset["symbol"]: asset
        for asset in original["portfolio"]["holdings"]
    }

    variant_holdings = {
        asset["symbol"]: asset
        for asset in variant["portfolio"]["holdings"]
    }

    violations = []

    for symbol in protected_symbols:
        if symbol not in original_holdings:
            raise ValueError(f"Protected asset {symbol} missing from original")

        original_quantity = original_holdings[symbol]["quantity"]
        new_quantity = (
            variant_holdings[symbol]["quantity"]
            if symbol in variant_holdings
            else 0
        )

        if new_quantity < original_quantity:
            violations.append({
                "restriction_type": "NO_SELL",
                "symbol": symbol,
                "original_quantity": original_quantity,
                "new_quantity": new_quantity
            })

    return violations


# Compare asset-class weights with allocation targets.
def check_allocation_violations(scenario):
    policy = scenario["investment_policy"]
    portfolio = scenario["portfolio"]

    targets = policy["target_allocations"]
    tolerance = policy["allocation_tolerance"]
    total = portfolio["total_value"]

    if total <= 0:
        raise ValueError("Portfolio total must be positive")

    allocations = {}

    for asset in portfolio["holdings"]:
        asset_class = asset["asset_class"]

        allocations[asset_class] = (
            allocations.get(asset_class, 0)
            + asset["market_value"]
        )

    allocations["CASH"] = (
        allocations.get("CASH", 0)
        + portfolio["cash_balance"]
    )

    violations = []

    for asset_class, target_pct in targets.items():
        actual_pct = allocations.get(asset_class, 0) / total

        if abs(actual_pct - target_pct) > tolerance + 1e-9:
            violations.append({
                "restriction_type": "ALLOCATION",
                "asset_class": asset_class,
                "target_pct": target_pct,
                "actual_pct": round(actual_pct, 4),
                "tolerance": tolerance
            })

    return violations



# Run the existing portfolio checks against a variant.
def verify_scenario(original, variant):
    validate_portfolio(variant["portfolio"])

    violations = []

    # Check individual stock limits
    violations.extend(
        check_individual_stock_limits(variant)
    )

    # Check protected assets
    violations.extend(
        check_no_sell_violations(original, variant)
    )

    # Check allocation constraints
    violations.extend(
        check_allocation_violations(variant)
    )

    return {
        "scenario_id": variant["scenario_id"],
        "violation_count": len(violations),
        "violations": violations
    }



# Generate the existing stock-limit strategy batch.
def generate_batch(scenario, symbol="AAPL"):
    strategies = ("below", "boundary", "above")
    results = []

    seen = {portfolio_signature(scenario)}

    for strategy in strategies:
        variant = generate_stock_limit_variant(
            scenario, symbol, strategy
        )

        signature = portfolio_signature(variant)

        if signature in seen:
            print(f"Skipping duplicate: {strategy}")
            continue

        seen.add(signature)

        scenario_number = scenario["scenario_id"][1:]
        variant_id = f"T{scenario_number}-v{len(results) + 1:03d}"
        variant["scenario_id"] = variant_id

        report = verify_scenario(scenario, variant)

        save_variant(variant, f"{variant_id}.json")
        save_variant(report, f"{variant_id}_report.json")

        results.append({
            "scenario_id": variant_id,
            "strategy": strategy,
            "violation_count": report["violation_count"]
        })

    return results



# Represent portfolio positions and cash for duplicate checks.
def portfolio_signature(scenario):
    portfolio = scenario["portfolio"]

    holdings = sorted(
        (
            asset["symbol"],
            asset["quantity"],
            asset["current_price"]
        )
        for asset in portfolio["holdings"]
    )

    return (
        tuple(holdings),
        portfolio["cash_balance"]
    )


# Apply multiple quantity changes to a copied scenario.
def generate_multi_asset_variant(scenario, changes):
    variant = create_variant(scenario)
    portfolio = variant["portfolio"]

    if not changes:
        raise ValueError("At least one asset change is required")

    holdings = {
        asset["symbol"]: asset
        for asset in portfolio["holdings"]
    }

    original_total = portfolio["total_value"]

    for symbol, new_quantity in changes.items():
        if symbol not in holdings:
            raise ValueError(f"Unknown asset: {symbol}")

        if type(new_quantity) is not int or new_quantity < 0:
            raise ValueError(f"Invalid quantity for {symbol}")

        asset = holdings[symbol]

        old_value = asset["market_value"]
        new_value = new_quantity * asset["current_price"]

        portfolio["cash_balance"] -= new_value - old_value
        asset["quantity"] = new_quantity

    if portfolio["cash_balance"] < 0:
        raise ValueError("Insufficient cash for asset changes")

    recalculate_portfolio(portfolio)
    validate_portfolio(portfolio)

    if abs(portfolio["total_value"] - original_total) > 0.01:
        raise ValueError("Portfolio total changed unexpectedly")

    return variant


# Try randomized holding changes until one is valid.
def generate_random_multi_asset_variant(
    scenario,
    seed=42,
    min_assets=2,
    max_assets=3,
    max_attempts=100
):
    rng = random.Random(seed)

    holdings = scenario["portfolio"]["holdings"]

    if len(holdings) < min_assets:
        raise ValueError("Not enough assets")

    for attempt in range(max_attempts):
        count = rng.randint(
            min_assets,
            min(max_assets, len(holdings))
        )

        selected = rng.sample(holdings, count)
        changes = {}

        for asset in selected:
            original_qty = asset["quantity"]

            change = rng.choice([-1, 1]) * rng.randint(1, 10)
            new_qty = max(0, original_qty + change)

            if new_qty != original_qty:
                changes[asset["symbol"]] = new_qty

        if not changes:
            continue

        try:
            variant = generate_multi_asset_variant(
                scenario, changes
            )
        except ValueError:
            continue

        if portfolio_signature(variant) == portfolio_signature(scenario):
            continue

        return variant, changes

    raise ValueError("Could not generate a valid variant")


# Search for a variant with a target violation count.
def generate_targeted_variant(
    scenario,
    target_violations,
    seed=42,
    max_attempts=500
):
    if type(target_violations) is not int or target_violations < 0:
        raise ValueError("Invalid target violation count")

    seen = {portfolio_signature(scenario)}

    for attempt in range(max_attempts):
        variant, changes = generate_random_multi_asset_variant(
            scenario,
            seed=seed + attempt
        )

        signature = portfolio_signature(variant)

        if signature in seen:
            continue

        seen.add(signature)

        variant["scenario_id"] = "TEST-TARGETED"

        report = verify_scenario(scenario, variant)

        if report["violation_count"] == target_violations:
            return variant, changes, report

    raise ValueError(
        f"Could not generate a variant with "
        f"{target_violations} violations"
    )



# Search for a variant with requested violation types.
def generate_targeted_types_variant(
    scenario,
    required_types,
    seed=42,
    max_attempts=500,
    exact_types=False
):
    required_types = set(required_types)

    if not required_types:
        raise ValueError("At least one violation type is required")

    supported_types = {
        "NO_SELL",
        "MAX_INDIVIDUAL_STOCK",
        "ALLOCATION"
    }

    if not required_types.issubset(supported_types):
        raise ValueError("Unsupported violation type requested")

    for attempt in range(max_attempts):
        try:
            variant, changes = generate_random_multi_asset_variant(
                scenario,
                seed=seed + attempt
            )
        except ValueError:
            continue

        variant["scenario_id"] = "TEST-TYPES"

        report = verify_scenario(scenario, variant)

        found_types = {
            "MAX_INDIVIDUAL_STOCK"
            if v["restriction_type"] == "MAX_INDIVIDUAL_STOCK"
            else v["restriction_type"]
            for v in report["violations"]
        }

        matches = (
            found_types == required_types
            if exact_types
            else required_types.issubset(found_types)
        )

        if matches:
            return variant, changes, report

    raise ValueError(
        f"Could not generate matching violation types: {required_types}"
    )


# Read the scenario answer type.
def get_scenario_type(scenario):
    answer_type = scenario.get("answer_type")

    if not answer_type:
        raise ValueError(
            f"Missing answer_type in {scenario.get('scenario_id')}"
        )

    return answer_type

# Inspect fields used to select a generation engine.
def get_scenario_capabilities(scenario):
    portfolio = scenario.get("portfolio")
    policy = scenario.get("investment_policy")

    return {
        "answer_type": get_scenario_type(scenario),
        "has_portfolio": isinstance(portfolio, dict),
        "has_holdings": (
            isinstance(portfolio, dict)
            and isinstance(portfolio.get("holdings"), list)
        ),
        "has_investment_policy": isinstance(policy, dict),
        "has_investment_options": bool(
            scenario.get("investment_options")
        ),
        "has_retirement_assumptions": bool(
            scenario.get("retirement_assumptions")
        )
    }


# Choose an engine label from the scenario capabilities.
def select_generation_engine(scenario):
    capabilities = get_scenario_capabilities(scenario)

    answer_type = capabilities["answer_type"]

    if answer_type == "VIOLATION_LIST":
        if (
            capabilities["has_holdings"]
            and capabilities["has_investment_policy"]
        ):
            return "PORTFOLIO_VIOLATIONS"

    if answer_type == "TRADE_PLAN":
        if (
            capabilities["has_holdings"]
            and capabilities["has_investment_policy"]
        ):
            return "PORTFOLIO_TRADES"

    if capabilities["has_investment_options"]:
        return "INVESTMENT_OPTIONS"

    if capabilities["has_retirement_assumptions"]:
        return "RETIREMENT_PROJECTION"

    return "UNSUPPORTED"


# Summarize engine selection for S01 through S10.
def inspect_all_scenarios():
    results = []

    for number in range(1, 11):
        scenario_id = f"S{number:02d}"

        try:
            scenario = load_scenario(scenario_id)

            results.append({
                "scenario_id": scenario_id,
                "answer_type": get_scenario_type(scenario),
                "engine": select_generation_engine(scenario)
            })

        except (FileNotFoundError, ValueError, KeyError) as error:
            results.append({
                "scenario_id": scenario_id,
                "error": str(error)
            })

    return results


# Ensure the selected engine matches the requested engine.
def validate_generation_engine(scenario, required_engine):
    selected_engine = select_generation_engine(scenario)

    if selected_engine != required_engine:
        raise ValueError(
            f"{scenario['scenario_id']} requires "
            f"{selected_engine}, not {required_engine}"
        )

    return True

# Build a violation report using verification_utils.
def generate_violation_report(scenario):
    from verification_utils import expected_violations

    violations = expected_violations(scenario)

    if violations is None:
        raise ValueError("Cannot calculate violations")

    return {
        "scenario_id": scenario["scenario_id"],
        "violation_count": len(violations),
        "violations": violations
    }

# Find the next unused variant filename identifier.
def next_variant_id(scenario_id):
    scenario_number = scenario_id[1:]
    version = 3 if scenario_id == "S03" else 1

    while True:
        variant_id = f"T{scenario_number}-v{version:03d}"

        scenario_file = OUTPUT_DIR / f"{variant_id}.json"
        report_file = OUTPUT_DIR / f"{variant_id}_report.json"

        if not scenario_file.exists() and not report_file.exists():
            return variant_id

        version += 1

# Compare against existing S03 variants only.
def is_duplicate_variant(variant):
    import copy

    candidate = copy.deepcopy(variant)
    candidate.pop("scenario_id", None)

    for filepath in OUTPUT_DIR.glob("T03-v*.json"):
        if filepath.stem.endswith("_report"):
            continue

        with open(filepath, "r") as file:
            existing = json.load(file)

        existing.pop("scenario_id", None)

        if existing == candidate:
            return True

    return False

# Generate one S03 holding-reduction variant and its report.
def generate_s03_variant(scenario, reduction=0.25, symbol="AAPL"):
    import copy

    variant = copy.deepcopy(scenario)
    variant_id = next_variant_id(scenario["scenario_id"])
    variant["scenario_id"] = variant_id

    # Reduce AAPL holding by 25% and transfer value to cash
    for holding in variant["portfolio"]["holdings"]:
        if holding["symbol"] == symbol:
            old_value = holding["market_value"]

            holding["quantity"] = round(
                holding["quantity"] * (1 - reduction)
            )
            holding["market_value"] = (
                holding["quantity"] * holding["current_price"]
            )

            difference = old_value - holding["market_value"]
            variant["portfolio"]["cash_balance"] += difference
            break

    # Recalculate portfolio total
    variant["portfolio"]["total_value"] = (
        variant["portfolio"]["cash_balance"]
        + sum(
            holding["market_value"]
            for holding in variant["portfolio"]["holdings"]
        )
    )

    validate_portfolio(variant["portfolio"])

    if is_duplicate_variant(variant):
        print(f"Skipped duplicate: {variant_id}")
        return None
    
    report = generate_violation_report(variant)

    save_variant(variant, f"{variant_id}.json")
    save_variant(report, f"{variant_id}_report.json")

    return report

# Generate the existing S03 reduction combinations.
def generate_s03_batch(scenario):
    reductions = [0.10, 0.25, 0.40]
    symbols = ["AAPL", "MSFT"]
    results = []

    for symbol in symbols:
        for reduction in reductions:
            report = generate_s03_variant(
                scenario,
                reduction=reduction,
                symbol=symbol
            )

            if report is None:
                continue

            results.append({
                "scenario_id": report["scenario_id"],
                "symbol": symbol,
                "reduction": reduction,
                "violation_count": report["violation_count"]
            })

    return results

# Dispatch only to generation paths implemented so far.
def generate_for_scenario(scenario):
    engine = select_generation_engine(scenario)

    if engine == "PORTFOLIO_TRADES":
        validate_generation_engine(
            scenario, "PORTFOLIO_TRADES"
        )

        variants = generate_batch(scenario)

        return {
            "scenario_id": scenario["scenario_id"],
            "engine": engine,
            "status": "GENERATED",
            "variants": variants
        }

    if engine == "PORTFOLIO_VIOLATIONS":
        return {
            "scenario_id": scenario["scenario_id"],
            "engine": engine,
            "status": "READY"
        }

    raise NotImplementedError(
        f"No generation engine for {scenario['scenario_id']}"
    )



# Check asset-class minimum and maximum policy ranges.
def check_class_range_restrictions(scenario):
    portfolio = scenario["portfolio"]
    total = portfolio["total_value"]

    class_values = {}

    for holding in portfolio["holdings"]:
        asset_class = holding["asset_class"]
        class_values[asset_class] = (
            class_values.get(asset_class, 0)
            + holding["market_value"]
        )

    class_values["CASH"] = (
        class_values.get("CASH", 0)
        + portfolio["cash_balance"]
    )

    violations = []

    for restriction in scenario["investment_policy"]["restrictions"]:
        if restriction["restriction_type"] != "CLASS_RANGE":
            continue

        params = restriction["parameters_json"]
        asset_class = params["asset_class"]

        actual_pct = class_values.get(asset_class, 0) / total
        min_pct = params["min_pct"]
        max_pct = params["max_pct"]

        if actual_pct < min_pct - 1e-9 or actual_pct > max_pct + 1e-9:
            violations.append({
                "type": "CLASS_RANGE",
                "asset_class": asset_class,
                "actual_pct": round(actual_pct, 6),
                "min_pct": min_pct,
                "max_pct": max_pct
            })

    return violations



# Check allocations to prohibited asset classes.
def check_prohibited_class_restrictions(scenario):
    portfolio = scenario["portfolio"]
    total = portfolio["total_value"]

    violations = []

    for restriction in scenario["investment_policy"]["restrictions"]:
        if restriction["restriction_type"] != "PROHIBITED_CLASS":
            continue

        params = restriction["parameters_json"]
        asset_class = params["asset_class"]
        max_pct = params.get("max_pct", 0)

        class_value = sum(
            holding["market_value"]
            for holding in portfolio["holdings"]
            if holding["asset_class"] == asset_class
        )

        actual_pct = class_value / total

        if actual_pct > max_pct + 1e-9:
            violations.append({
                "type": "PROHIBITED_CLASS",
                "asset_class": asset_class,
                "actual_pct": round(actual_pct, 6),
                "max_pct": max_pct
            })

    return violations


# Combine the existing class-level policy checks.
def verify_portfolio_restrictions(scenario):
    violations = []

    violations.extend(
        check_class_range_restrictions(scenario)
    )

    violations.extend(
        check_prohibited_class_restrictions(scenario)
    )

    return {
        "scenario_id": scenario["scenario_id"],
        "violation_count": len(violations),
        "violations": violations
    }


# Main entry point: inspect the configured scenario without writing files.
# Generation helpers above can be called explicitly when needed.
if __name__ == "__main__":
    print("Starting scenario generator...")

    scenario = load_scenario(SCENARIO_ID)
    scenario_type = get_scenario_type(scenario)
    capabilities = get_scenario_capabilities(scenario)
    engine = select_generation_engine(scenario)

    print("Scenario ID:", scenario["scenario_id"])
    print("Answer type:", scenario_type)
    print("Scenario capabilities:", json.dumps(capabilities, indent=2))
    print("Selected generation engine:", engine)

    # No variants are generated or saved by this entry point.
