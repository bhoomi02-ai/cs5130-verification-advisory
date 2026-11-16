"""Parse a verifier's reply into a Verdict.

The verifier replies with one JSON object (see prompts.REVIEW_INSTRUCTIONS):
{"verdict": "accept" | "reject" | "abstain", "issues": [...], "reason": "..."}.
A reply that cannot be read raises ValueError; the pipeline should then store
the verdict as missing (None), which metrics counts in n_missing_verdict,
rather than guess one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from advisory.prompts import VERDICTS

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@dataclass
class Verdict:
    verdict: str  # one of VERDICTS
    reason: str = ""
    issues: list[str] = field(default_factory=list)


def parse_verdict(text: str) -> Verdict:
    """Read the verifier's JSON reply. Raises ValueError if it is not a valid verdict."""
    match = _FENCE.match(text or "")
    try:
        data = json.loads(match.group(1) if match else text)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"verdict is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("verdict is not a JSON object")

    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {data.get('verdict')!r}")

    issues = data.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]
    if not isinstance(issues, list):
        raise ValueError("issues must be a list of strings")
    issues = [str(issue).strip() for issue in issues if str(issue).strip()]

    reason = data.get("reason") or ""
    return Verdict(verdict=verdict, reason=str(reason).strip(), issues=issues)
