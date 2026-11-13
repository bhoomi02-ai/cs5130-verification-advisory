"""The seeded catalog rebuilds cleanly and stays away from the agents."""

import ast
import json
from pathlib import Path

from advisory.checker import check
from advisory.inject import build_catalog

_ROOT = Path(__file__).resolve().parents[1]
_AGENTS = ("conditions.py", "llm.py")
_ERROR_COUNTS = {
    "policy_violation": 3,
    "arithmetic": 3,
    "grounding": 3,
    "unit": 3,
    "wrong_status": 3,
    "violation_list": 2,
}


def test_agents_do_not_import_reference_answers():
    for name in _AGENTS:
        tree = ast.parse((_ROOT / "src" / "advisory" / name).read_text(encoding="utf-8"))
        assert not _imports_reference(tree), name


def test_catalog_rebuilds_and_every_file_passes_its_gate():
    written = build_catalog(seed=0)
    saved = sorted((_ROOT / "data" / "injected").glob("*.json"))
    assert sorted(path.name for path in written) == [path.name for path in saved]
    records = [json.loads(path.read_text(encoding="utf-8")) for path in saved]
    controls = [record for record in records if record["error_type"] == "none"]
    errors = [record for record in records if record["error_type"] != "none"]
    assert len(controls) == 14
    assert {record["scenario_id"] for record in controls} == {f"S{index:02d}" for index in range(1, 11)}
    assert sum(record["description"] == "Alternative valid trade plan." for record in controls) == 4
    assert {record["error_type"]: 0 for record in errors} | {
        error_type: sum(record["error_type"] == error_type for record in errors)
        for error_type in _ERROR_COUNTS
    } == _ERROR_COUNTS
    for record in records:
        _assert_gate(record)


def _assert_gate(record: dict):
    result = check(record["scenario_id"], record["response"])
    assert result["format_failure"] is False
    if record["error_type"] == "none":
        assert record["expected_correct"] is True
        assert result["correct"] is True
        return
    failed = [row for row in result["checks"] if not row["passed"]]
    named = [row for row in failed if row["check_type"] == record["expected_check"]]
    assert len(named) == 1
    assert named[0]["details"]["reason"] == record["expected_reason"]
    also_fails = [
        {"check": row["check_type"], "reason": row["details"].get("reason")}
        for row in failed
        if row["check_type"] != record["expected_check"]
    ]
    assert record["also_fails"] == also_fails


def _imports_reference(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [module, *[f"{module}.{alias.name}" for alias in node.names]]
        else:
            continue
        for name in names:
            if name == "advisory.evaluation.reference" or name.startswith("advisory.evaluation.reference."):
                return True
    return False
