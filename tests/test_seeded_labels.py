"""The seeded-error labels stay in sync with the seeded set (data/injected/)."""

import csv
import json
from pathlib import Path

import pytest

from advisory.checker import check

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "evaluation" / "seeded_error_labels.csv"
INJECTED = ROOT / "data" / "injected"


def _labels():
    with LABELS.open(newline="") as f:
        return list(csv.DictReader(f))


def _seeded_errors():
    files = [p for p in sorted(INJECTED.glob("*.json")) if json.loads(p.read_text())["error_type"] != "none"]
    return {p.stem: json.loads(p.read_text()) for p in files}


def test_one_label_per_seeded_error():
    labels = [row["answer_id"] for row in _labels()]
    assert len(labels) == len(set(labels))
    assert set(labels) == set(_seeded_errors())  # every seeded error labelled, no controls, nothing extra


def test_every_label_is_complete():
    for row in _labels():
        for column in ("field", "wrong_value", "correct_value", "match_terms"):
            assert row[column].strip(), (row["answer_id"], column)


@pytest.mark.parametrize("row", _labels(), ids=lambda r: r["answer_id"])
def test_labelled_wrong_value_is_what_the_checker_flags(row):
    """For reported figures, the label's wrong and correct values match the checker's observed and expected."""
    seeded = _seeded_errors()[row["answer_id"]]
    result = check(seeded["scenario_id"], seeded["response"])
    assert not result["correct"]
    if not row["field"].startswith("metrics:"):
        return
    items = [item for c in result["checks"] if not c["passed"] for item in (c.get("details") or {}).get("items", [])]
    observed = {round(float(i["observed"]), 2) for i in items if "observed" in i and isinstance(i["observed"], (int, float))}
    expected = {round(float(i["expected"]), 2) for i in items if "expected" in i and isinstance(i["expected"], (int, float))}
    assert round(float(row["wrong_value"]), 2) in observed, (row["answer_id"], observed)
    assert round(float(row["correct_value"]), 2) in expected, (row["answer_id"], expected)


def test_labels_live_with_the_evaluation_files():
    # The labels are an answer key: they belong in data/evaluation/ (scorer only), never next to agent inputs.
    assert LABELS.parent.name == "evaluation"
    assert not list(INJECTED.glob("*label*"))
