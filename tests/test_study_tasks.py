"""The main study's task list: 32 tasks, each with a scenario and an answer key."""

import json

import pytest

from advisory import pipeline

TASKS = pipeline.read_task_file()
EVALUATION = pipeline.ROOT / "data" / "evaluation"
# Same numbers and answers as their base scenario, so they must stay out.
DUPLICATES = {"T03-v005": "S03", "T04-v001": "S04", "T06-v001": "S06"}


def test_thirty_two_tasks_no_duplicates():
    assert len(TASKS) == 32 == len(set(TASKS))


def test_every_family_is_covered():
    families = {"S" + t[1:3] if t.startswith("T") else t for t in TASKS}
    assert families == {f"S{i:02d}" for i in range(1, 11)}
    for i in range(1, 9):
        assert sum(t.startswith(f"T{i:02d}-") for t in TASKS) == 2, i
    assert sum(t[:3] in ("T09", "T10") for t in TASKS) == 6


@pytest.mark.parametrize("name", ["ground_truth.json", "checks.json"])
def test_every_task_has_a_scenario_and_an_answer_key(name):
    key = json.loads((EVALUATION / name).read_text(encoding="utf-8"))
    for task in TASKS:
        assert (pipeline.SCENARIO_DIR / f"{task}.json").exists(), task
        assert task in key, (task, name)


def test_copies_of_base_scenarios_are_left_out():
    assert not set(DUPLICATES) & set(TASKS)


def test_read_task_file_skips_comments_and_refuses_repeats(tmp_path):
    f = tmp_path / "tasks.txt"
    f.write_text("# header\nS01\n\nS02  # note\n")
    assert pipeline.read_task_file(f) == ["S01", "S02"]
    f.write_text("S01\nS01\n")
    with pytest.raises(ValueError, match="more than once"):
        pipeline.read_task_file(f)


def _script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_condition", pipeline.ROOT / "scripts" / "run_condition.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_condition_reads_the_task_file(monkeypatch, tmp_path):
    script = _script()
    f = tmp_path / "tasks.txt"
    f.write_text("S01\nS02\n")
    ran = []
    monkeypatch.setattr(script, "RESULTS_PATH", tmp_path / "results.jsonl")
    monkeypatch.setattr(script, "run_one", lambda c, t, r, s, results_path: ran.append(t) or {
        "status": "unchecked", "initial_correct": True, "final_correct": True,
        "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})
    assert script.main(["--condition", "none", "--model", "m", "--tasks-file", str(f)]) == 0
    assert ran == ["S01", "S02"]


def test_run_condition_refuses_both_task_options(tmp_path):
    with pytest.raises(SystemExit):
        _script().main(["--condition", "none", "--model", "m", "--tasks", "S01", "--tasks-file", str(tmp_path / "x")])
