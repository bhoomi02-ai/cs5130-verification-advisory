"""status_correct: whether the checker's STATUS check passed (status-only accuracy, e.g. S09/S10).

The first test uses the real checker on the hand-made answers in data/injected.
"""

import json

import pytest

from advisory import llm, pipeline
from advisory.drafts import Settings

SETTINGS = Settings("gemini-3.1-flash-lite")
INJECTED = pipeline.ROOT / "data" / "injected"


def answer(name):
    return json.dumps(json.loads((INJECTED / name).read_text())["response"])


@pytest.mark.parametrize(
    "name, correct, status_ok",
    [
        ("S09_none_reference.json", True, True),
        ("S09_wrong_status_feasible.json", False, False),
        ("S10_wrong_status_feasible.json", False, False),
        ("S01_policy_violation_sell_aapl.json", False, True),  # wrong answer, right status
    ],
)
def test_status_correct_comes_from_the_checker(monkeypatch, tmp_path, name, correct, status_ok):
    monkeypatch.setattr(llm, "call_model", lambda **kwargs: answer(name))
    row = pipeline.run_one(
        "none", name[:3], 1, SETTINGS,
        results_path=tmp_path / "r.jsonl", draft_dir=tmp_path / "d", log_path=tmp_path / "l.jsonl",
    )
    assert (row["initial_correct"], row["initial_status_correct"]) == (correct, status_ok)
    assert row["final_status_correct"] == status_ok  # none: the final answer is the draft


def test_format_failure_counts_as_wrong_status():
    graded = {"correct": False, "format_failure": True, "checks": [{"check_type": "SCHEMA", "passed": False}]}
    assert pipeline.status_correct(graded) is False


def test_no_status_check_gives_none():
    graded = {"correct": True, "format_failure": False, "checks": [{"check_type": "SCHEMA", "passed": True}]}
    assert pipeline.status_correct(graded) is None
