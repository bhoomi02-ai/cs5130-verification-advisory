
"""Standalone portfolio verification utilities.

Adapted from the team's deterministic checker.
"""

def expected_violations(scenario):
    portfolio = scenario["portfolio"]
    total = portfolio["total_value"]
    cash = portfolio["cash_balance"]

    if total is None or total <= 0 or cash is None:
        return None

    positions = []

    for holding in portfolio.get("holdings", []):
        value = holding.get("market_value")

        if value is None:
            quantity = holding.get("quantity")
            price = holding.get("current_price")

            if quantity is None or price is None:
                return None

            value = quantity * price

        positions.append({
            "symbol": holding["symbol"],
            "asset_class": holding.get("asset_class"),
            "asset_type": holding.get("asset_type"),
            "value": float(value)
        })

    class_values = {}

    for position in positions:
        asset_class = position["asset_class"]
        class_values[asset_class] = (
            class_values.get(asset_class, 0)
            + position["value"]
        )

    class_values["CASH"] = (
        class_values.get("CASH", 0)
        + cash
    )

    violations = []

    for restriction in scenario["investment_policy"].get("restrictions", []):
        kind = restriction["restriction_type"]
        params = restriction.get("parameters_json") or {}

        if kind == "CLASS_RANGE":
            asset_class = params["asset_class"]
            actual = class_values.get(asset_class, 0) / total

            minimum = params.get("min_pct")
            maximum = params.get("max_pct")

            if minimum is not None and actual < minimum - 1e-12:
                violations.append({
                    "kind": "CLASS_MIN",
                    "scope": asset_class,
                    "observed": actual,
                    "limit": minimum
                })

            if maximum is not None and actual > maximum + 1e-12:
                violations.append({
                    "kind": "CLASS_MAX",
                    "scope": asset_class,
                    "observed": actual,
                    "limit": maximum
                })

        elif kind == "MAX_INDIVIDUAL_STOCK":
            limit = params["max_pct"]

            for position in positions:
                if position["asset_type"] != "INDIVIDUAL_STOCK":
                    continue

                actual = position["value"] / total

                if actual > limit + 1e-12:
                    violations.append({
                        "kind": "INDIVIDUAL_MAX",
                        "scope": position["symbol"],
                        "observed": actual,
                        "limit": limit
                    })

        elif kind == "PROHIBITED_CLASS":
            asset_class = params["asset_class"]
            limit = params.get("max_pct", 0)
            actual = class_values.get(asset_class, 0) / total

            if actual > limit + 1e-12:
                violations.append({
                    "kind": "PROHIBITED_CLASS",
                    "scope": asset_class,
                    "observed": actual,
                    "limit": limit
                })

    return violations
