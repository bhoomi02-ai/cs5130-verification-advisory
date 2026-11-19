"""Right-reason scoring for seeded-error verifier reviews.

Scorer-side only: agents never import this module or the label CSV. Reviews in
``results/seeded/`` are classified against ``seeded_error_labels.csv``.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = ROOT / "data" / "evaluation" / "seeded_error_labels.csv"
SEEDED_DIR = ROOT / "results" / "seeded"

# Comma/space thousands: \d{1,3}(?:[ ,]\d{3})+ with optional decimals.
# Space groups are accepted only when unambiguous (see _grouped_number_ok).
_GROUPED_NUMBER = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<token>-?\s*\$?\s*\d{1,3}(?:[ ,]\d{3})+(?:\.\d+)?\s*%?)"
    r"(?![A-Za-z0-9])"
)
_PLAIN_NUMBER = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<token>-?\s*\$?\s*\d+(?:\.\d+)?\s*%?)"
    r"(?![A-Za-z0-9])"
)
_NON_NUMERIC_FIELD = re.compile(
    r"^(feasible|infeasible|violations_found|missing|listed\b)",
    re.IGNORECASE,
)

__all__ = [
    "LABELS_PATH",
    "SEEDED_DIR",
    "load_labels",
    "extract_numbers",
    "classify_review",
    "right_reason_metrics",
    "load_seeded_records",
    "audit_table",
]


def load_labels(path: Path | str = LABELS_PATH) -> list[dict]:
    """Load seeded-error labels and parse numeric targets.

    Parameters
    ----------
    path : path-like, optional
        CSV with columns ``answer_id``, ``field``, ``wrong_value``,
        ``correct_value``, ``also_wrong``, ``match_terms``.

    Returns
    -------
    list of dict
        One dict per row. Adds ``wrong_numbers``, ``correct_numbers``,
        ``also_wrong_numbers`` (floats) and ``match_terms`` as a list of
        strings. Non-numeric fields (``FEASIBLE``, ``missing``,
        ``listed (...)``) contribute no numbers.
    """
    path = Path(path)
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            terms = [
                term.strip()
                for term in (row.get("match_terms") or "").split(";")
                if term.strip()
            ]
            rows.append(
                {
                    "answer_id": row["answer_id"],
                    "field": row.get("field") or "",
                    "wrong_value": row.get("wrong_value") or "",
                    "correct_value": row.get("correct_value") or "",
                    "also_wrong": row.get("also_wrong") or "",
                    "match_terms": terms,
                    # Leading figure only for wrong/correct (e.g. "-1 (sells 1 AAPL)" → -1).
                    "wrong_numbers": _primary_field_numbers(row.get("wrong_value") or ""),
                    "correct_numbers": _primary_field_numbers(row.get("correct_value") or ""),
                    # also_wrong keeps every figure, including those in parentheses.
                    "also_wrong_numbers": _field_numbers(row.get("also_wrong") or ""),
                }
            )
    return rows


def extract_numbers(text: str) -> list[float]:
    """Return every number mentioned in ``text``.

    Parameters
    ----------
    text : str
        Free text. ``$``, commas and spaces inside a *thousands-grouped*
        number are ignored. A space or comma joins digits only for proper
        thousands groups (``\\d{1,3}(?:[ ,]\\d{3})+``); adjacent small
        figures such as ``\"3 100\"`` or a date ``\"2026 11 23\"`` stay
        separate. A token ending in ``%`` yields both the raw figure and
        its fraction (``\"35%\"`` → ``35`` and ``0.35``).

    Returns
    -------
    list of float
        Numbers in left-to-right order (percentage fractions follow the raw
        value).
    """
    if not text:
        return []
    spans: list[tuple[int, int, str]] = []
    for match in _GROUPED_NUMBER.finditer(text):
        token = match.group("token")
        if _grouped_number_ok(token):
            spans.append((match.start(), match.end(), token))
    for match in _PLAIN_NUMBER.finditer(text):
        start, end = match.start(), match.end()
        if any(not (end <= left or start >= right) for left, right, _ in spans):
            continue
        spans.append((start, end, match.group("token")))
    spans.sort(key=lambda item: item[0])

    found: list[float] = []
    for _, _, token in spans:
        is_percent = token.rstrip().endswith("%")
        cleaned = token.replace("$", "").replace(",", "").replace("%", "")
        cleaned = re.sub(r"\s+", "", cleaned)
        if cleaned in ("", "-", ".", "-."):
            continue
        try:
            value = float(cleaned)
        except ValueError:
            continue
        found.append(value)
        if is_percent:
            found.append(value / 100.0)
    return found


def classify_review(review: dict, label: dict) -> dict:
    """Classify one verifier review against a seeded-error label.

    Parameters
    ----------
    review : dict
        A single review entry with ``verdict``, ``issues`` and ``reason``.
    label : dict
        A row from ``load_labels`` for the same ``answer_id``.

    Returns
    -------
    dict
        ``decision`` is one of ``right_reason``, ``consequence_only``,
        ``wrong_reason``, ``abstained``, ``not_flagged``, ``missing_verdict``.
        ``matched_in`` is ``\"issues\"``, ``\"reason\"`` or ``None``.
        ``matched`` lists the numbers or terms that justified the decision.

    Notes
    -----
    Verdict ``abstain`` is always ``abstained`` (issues/reason are ignored).
    A matching wrong/correct number alone counts as ``right_reason`` only when
    ``abs(label_value) >= 100``. Smaller values also need a non-numeric
    ``match_terms`` hit in the same text. Two distinct match_terms still
    suffice on their own. ``also_wrong`` numbers for ``consequence_only``
    follow the same size rule.
    """
    verdict = review.get("verdict")
    if verdict is None:
        return {"decision": "missing_verdict", "matched_in": None, "matched": []}
    if verdict == "accept":
        return {"decision": "not_flagged", "matched_in": None, "matched": []}
    if verdict == "abstain":
        return {"decision": "abstained", "matched_in": None, "matched": []}
    if verdict != "reject":
        return {"decision": "missing_verdict", "matched_in": None, "matched": []}

    issues = review.get("issues") or []
    if isinstance(issues, str):
        issues_text = issues
    else:
        issues_text = "\n".join(str(item) for item in issues)
    reason_text = str(review.get("reason") or "")

    for source, text in (("issues", issues_text), ("reason", reason_text)):
        matched = _right_reason_hits(text, label)
        if matched:
            return {"decision": "right_reason", "matched_in": source, "matched": matched}

    for source, text in (("issues", issues_text), ("reason", reason_text)):
        matched = _consequence_hits(text, label)
        if matched:
            return {
                "decision": "consequence_only",
                "matched_in": source,
                "matched": matched,
            }

    return {"decision": "wrong_reason", "matched_in": None, "matched": []}


def right_reason_metrics(records: list[dict], labels: list[dict]) -> pd.DataFrame:
    """Aggregate right-reason rates per condition and verifier model.

    Parameters
    ----------
    records : list of dict
        Seeded review files as loaded by ``load_seeded_records``.
    labels : list of dict
        Output of ``load_labels``.

    Returns
    -------
    DataFrame
        One row per ``(condition, verifier_model)`` with counts and rates.
        Records with ``status == \"failed\"`` are left out of every rate and
        counted in ``n_failed`` (same pattern as ``metrics.summarize``).
        ``abstain`` counts as flagged for ``seeded_catch_rate`` /
        ``seeded_false_alarm_rate``, but ``right_reason_rate`` uses only
        ``right_reason / (right_reason + consequence_only + wrong_reason)``.
        ``right_reason_catch_rate`` still divides by errors with a readable
        verdict. Rates are ``None`` when their denominator is zero.

    Raises
    ------
    ValueError
        If a record has a non-``none`` ``error_type`` but no label row.
    """
    label_by_id = {row["answer_id"]: row for row in labels}
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record.get("condition"), record.get("verifier_model"))
        groups.setdefault(key, []).append(record)

    rows: list[dict] = []
    for (condition, verifier_model), group in sorted(
        groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        n_errors = 0
        n_controls = 0
        n_missing = 0
        n_failed = 0
        flagged_errors = 0
        errors_with_verdict = 0
        flagged_controls = 0
        controls_with_verdict = 0
        n_right = 0
        n_consequence = 0
        n_wrong = 0
        n_abstained = 0

        for record in group:
            if record.get("status") == "failed":
                n_failed += 1
                continue
            kind = _record_kind(record, label_by_id)
            review = _first_review(record)
            verdict = review.get("verdict")
            if kind == "control":
                n_controls += 1
                if verdict is None:
                    n_missing += 1
                else:
                    controls_with_verdict += 1
                    if verdict in ("reject", "abstain"):
                        flagged_controls += 1
                continue

            n_errors += 1
            classified = classify_review(review, label_by_id[record["answer_id"]])
            decision = classified["decision"]
            if decision == "missing_verdict":
                n_missing += 1
                continue
            errors_with_verdict += 1
            if decision == "not_flagged":
                continue
            flagged_errors += 1
            if decision == "right_reason":
                n_right += 1
            elif decision == "consequence_only":
                n_consequence += 1
            elif decision == "abstained":
                n_abstained += 1
            else:
                n_wrong += 1

        reason_den = n_right + n_consequence + n_wrong
        rows.append(
            {
                "condition": condition,
                "verifier_model": verifier_model,
                "n_errors": n_errors,
                "n_controls": n_controls,
                "n_failed": n_failed,
                "n_missing_verdict": n_missing,
                "seeded_catch_rate": _rate(flagged_errors, errors_with_verdict),
                "seeded_false_alarm_rate": _rate(flagged_controls, controls_with_verdict),
                "right_reason": n_right,
                "consequence_only": n_consequence,
                "wrong_reason": n_wrong,
                "abstained": n_abstained,
                "right_reason_rate": _rate(n_right, reason_den),
                "right_reason_catch_rate": _rate(n_right, errors_with_verdict),
            }
        )
    return pd.DataFrame(rows)


def load_seeded_records(directory: Path | str = SEEDED_DIR) -> list[dict]:
    """Load every seeded review JSON from ``directory``.

    Parameters
    ----------
    directory : path-like, optional
        Directory of ``*.json`` review files.

    Returns
    -------
    list of dict
        Parsed records, sorted by file name.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    records: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def audit_table(records: list[dict], labels: list[dict]) -> pd.DataFrame:
    """One audit row per seeded review record.

    Parameters
    ----------
    records : list of dict
        Seeded review files.
    labels : list of dict
        Output of ``load_labels``.

    Returns
    -------
    DataFrame
        Columns ``answer_id``, ``condition``, ``verifier_model``, ``verdict``,
        ``decision``, ``matched_in``, ``matched``, ``issues``.
    """
    label_by_id = {row["answer_id"]: row for row in labels}
    rows: list[dict] = []
    for record in records:
        review = _first_review(record)
        issues = review.get("issues") or []
        if isinstance(issues, str):
            issues_text = issues
        else:
            issues_text = " | ".join(str(item) for item in issues)
        if record.get("status") == "failed":
            decision, matched_in, matched = "failed", None, []
        else:
            kind = _record_kind(record, label_by_id)
            if kind == "control":
                verdict = review.get("verdict")
                if verdict is None:
                    decision, matched_in, matched = "missing_verdict", None, []
                elif verdict == "accept":
                    decision, matched_in, matched = "not_flagged", None, []
                elif verdict in ("reject", "abstain"):
                    decision, matched_in, matched = "false_alarm", None, []
                else:
                    decision, matched_in, matched = "missing_verdict", None, []
            else:
                classified = classify_review(review, label_by_id[record["answer_id"]])
                decision = classified["decision"]
                matched_in = classified["matched_in"]
                matched = classified["matched"]
        rows.append(
            {
                "answer_id": record.get("answer_id"),
                "condition": record.get("condition"),
                "verifier_model": record.get("verifier_model"),
                "verdict": review.get("verdict"),
                "decision": decision,
                "matched_in": matched_in,
                "matched": matched,
                "issues": issues_text,
            }
        )
    return pd.DataFrame(rows)


def _grouped_number_ok(token: str) -> bool:
    """Whether a ``\\d{1,3}(?:[ ,]\\d{3})+`` match is a real thousands group.

    Comma groups are always kept. Space groups are kept when there is a
    decimal part, at least two trailing ``\\d{3}`` groups, or the leading
    group already has three digits — so ``\"137 688.62\"`` stays one number
    while ``\"3 100\"`` and ``\"2026 11 23\"`` do not.
    """
    cleaned = token.replace("$", "").replace("%", "").strip()
    cleaned = re.sub(r"^-", "", cleaned).strip()
    if "," in cleaned:
        return True
    if "." in cleaned:
        body, _, _decimals = cleaned.partition(".")
    else:
        body = cleaned
    parts = body.split()
    if len(parts) >= 3:
        return True
    if parts and len(parts[0]) == 3:
        return True
    return False


def _field_numbers(text: str) -> list[float]:
    text = (text or "").strip()
    if not text or _NON_NUMERIC_FIELD.match(text):
        return []
    return extract_numbers(text)


def _primary_field_numbers(text: str) -> list[float]:
    """Numbers from the main field text, ignoring a trailing parenthetical."""
    text = (text or "").strip()
    if not text or _NON_NUMERIC_FIELD.match(text):
        return []
    head = re.split(r"\s*\(", text, maxsplit=1)[0]
    return extract_numbers(head) or extract_numbers(text)


def _first_review(record: dict) -> dict:
    reviews = record.get("reviews") or []
    for review in reviews:
        if review.get("stage") == "first":
            return review
    if reviews:
        return reviews[0]
    return {"verdict": None, "issues": [], "reason": ""}


def _record_kind(record: dict, label_by_id: dict) -> str:
    if record.get("is_control") is True or record.get("error_type") == "none":
        return "control"
    answer_id = record.get("answer_id")
    if answer_id in label_by_id:
        return "error"
    error_type = record.get("error_type")
    if error_type is not None and error_type != "none":
        raise ValueError(
            f"missing label for {answer_id!r} (error_type={error_type!r})"
        )
    return "control"


def _right_reason_hits(text: str, label: dict) -> list:
    if not text:
        return []
    targets = list(label.get("wrong_numbers") or []) + list(
        label.get("correct_numbers") or []
    )
    lexical_terms = _non_numeric_terms(label.get("match_terms") or [])
    has_lexical = any(_term_in_text(term, text) for term in lexical_terms)

    number_hits: list[float] = []
    seen_numbers: set[float] = set()
    for number in extract_numbers(text):
        for target in targets:
            if not _numbers_match(number, target):
                continue
            if abs(target) < 100 and not has_lexical:
                continue
            key = round(number, 10)
            if key not in seen_numbers:
                seen_numbers.add(key)
                number_hits.append(number)
            break
    if number_hits:
        return number_hits

    term_hits = [
        term for term in label.get("match_terms") or [] if _term_in_text(term, text)
    ]
    distinct = []
    seen_terms: set[str] = set()
    for term in term_hits:
        key = term.lower()
        if key not in seen_terms:
            seen_terms.add(key)
            distinct.append(term)
    if len(distinct) >= 2:
        return distinct
    return []


def _consequence_hits(text: str, label: dict) -> list:
    targets = list(label.get("also_wrong_numbers") or [])
    if not text or not targets:
        return []
    lexical_terms = _non_numeric_terms(label.get("match_terms") or [])
    has_lexical = any(_term_in_text(term, text) for term in lexical_terms)
    matched: list = []
    seen: set = set()
    for number in extract_numbers(text):
        for target in targets:
            if not _numbers_match(number, target):
                continue
            if abs(target) < 100 and not has_lexical:
                continue
            key = round(number, 10)
            if key not in seen:
                seen.add(key)
                matched.append(number)
            break
    return matched


def _numbers_match(observed: float, target: float) -> bool:
    tolerance = 1.0 if abs(target) >= 1 else 0.005
    return abs(observed - target) <= tolerance


def _term_is_numeric(term: str) -> bool:
    cleaned = term.replace("$", "").replace(",", "").replace("%", "").strip()
    if not cleaned:
        return False
    try:
        float(cleaned)
    except ValueError:
        return False
    return True


def _non_numeric_terms(terms: list[str]) -> list[str]:
    return [term for term in terms if not _term_is_numeric(term)]


def _normalize_term_text(text: str) -> str:
    """Map common no-sell paraphrases onto the ``no-sell`` match term."""
    text_l = text.lower()
    text_l = re.sub(r"\bmay not be sold\b", "no-sell", text_l)
    text_l = re.sub(r"\bnot be sold\b", "no-sell", text_l)
    text_l = re.sub(r"\bdo not sell\b", "no-sell", text_l)
    return text_l


def _term_in_text(term: str, text: str) -> bool:
    if not term or not text:
        return False
    term_l = term.lower()
    text_l = _normalize_term_text(text)
    escaped = re.escape(term_l)
    if re.fullmatch(r"[a-z]+", term_l):
        # Prefix inflection so "sell" matches "selling".
        pattern = rf"(?<![a-z0-9]){escaped}[a-z]*(?![a-z0-9])"
    elif re.match(r"^[a-z0-9]", term_l) and re.search(r"[a-z0-9]$", term_l):
        pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    else:
        pattern = escaped
    return re.search(pattern, text_l) is not None


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
