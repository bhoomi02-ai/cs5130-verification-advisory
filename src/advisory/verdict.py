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

_OPENING = re.compile(r"^\s*(?:```(?:json)?\s*)?", re.IGNORECASE)
_CLOSING_FENCE = re.compile(r"^\s*```")


def split_leading_json(text: str):
    """(the JSON value at the start of `text`, the text after it), ignoring an optional code fence.

    The returned trailing text has the closing fence removed and is stripped,
    so it is "" when nothing follows the JSON. Raises ValueError if the text
    does not start with JSON.
    """
    text = text or ""
    rest = text[_OPENING.match(text).end():]
    try:
        value, end = json.JSONDecoder().raw_decode(rest)
    except json.JSONDecodeError as e:
        raise ValueError(f"reply does not start with JSON: {e}") from e
    trailing = rest[end:]
    fence = _CLOSING_FENCE.match(trailing)
    if fence:
        trailing = trailing[fence.end():]
    return value, trailing.strip()


def leading_json(text: str):
    """The JSON value at the start of `text` (after an optional code fence), ignoring what follows.

    Some models keep writing after their JSON, for example commentary or, after
    a calculator request, results they made up. Only the leading object is the
    model's actual reply; the rest is ignored. Raises ValueError if the text
    does not start with JSON.
    """
    return split_leading_json(text)[0]


@dataclass
class Verdict:
    verdict: str  # one of VERDICTS
    reason: str = ""
    issues: list[str] = field(default_factory=list)
    # Text the reply had after its JSON, which was ignored. Kept so replies whose
    # commentary contradicts the verdict can be audited; not part of equality.
    trailing: str = field(default="", compare=False)


def parse_verdict(text: str) -> Verdict:
    """Read the verifier's JSON reply. Raises ValueError if it is not a valid verdict.

    The verdict is the JSON object the reply starts with; text after it is ignored.
    """
    try:
        data, trailing = split_leading_json(text)
    except ValueError as e:
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
    return Verdict(verdict=verdict, reason=str(reason).strip(), issues=issues, trailing=trailing)
