"""Shared S06-family helpers for checker and regrade tests."""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_GROUND_TRUTH = json.loads((_ROOT / "data" / "evaluation" / "ground_truth.json").read_text())

S06_FAMILY = ("S06", "T06-v001", "T06-v002", "T06-v003", "T06-v004")

# Still recorded in ground_truth.json, but no longer a required/graded metric.
S06_OPTIONAL_GROUND_TRUTH_KEYS = frozenset({"additional_monthly_contribution"})


def s06_family_rounded_only(scenario_id: str, *, cents=None, wrong_cents: bool = False):
    """Answer that reports only the upward-cent payment (no unrounded PAYMENT)."""
    ground_truth = _GROUND_TRUTH[scenario_id]
    metrics = ground_truth["metrics"]
    payment_cents = metrics["additional_monthly_contribution_cents"]
    if wrong_cents:
        payment_cents = round(payment_cents + 1.0, 2)
    elif cents is not None:
        payment_cents = cents

    def _money(subject, stage, value, measure="VALUE", rounding="NONE"):
        return {
            "subject": subject,
            "measure": measure,
            "stage": stage,
            "rounding": rounding,
            "unit": "MONEY",
            "value": value,
        }

    return {
        "scenario_id": ground_truth["scenario_id"],
        "answer_type": ground_truth["answer_type"],
        "status": ground_truth["status"],
        "trades": ground_truth["trades"],
        "metrics": {
            "observations": [
                _money("PORTFOLIO", "FINAL", metrics["final_balance"]),
                _money(
                    "PORTFOLIO",
                    "FINAL",
                    metrics["target_shortfall"],
                    measure="DIFFERENCE",
                ),
                _money(
                    "PORTFOLIO",
                    "FINAL",
                    payment_cents,
                    measure="PAYMENT",
                    rounding="UPWARD_CENT",
                ),
            ],
            "selection": None,
        },
        "violations": [],
        "explanation": "plan",
    }
