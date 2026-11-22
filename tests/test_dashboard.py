"""Dashboard data functions, and a smoke test of the Streamlit app (skipped without streamlit).

Uses a small results folder written into tmp_path; no model calls.
"""

import json

import pytest

from advisory import dashboard_data as data

TASKS = ["t1", "t2"]


def _row(task, condition, run, initial, final, status=None, verdict=None):
    verified = condition != "none"
    return {
        "run_id": f"{condition}-m-{task}-r{run}",
        "draft_run_id": f"draft-m-{task}-r{run}",
        "task_id": task,
        "condition": condition,
        "run_number": run,
        "model": "m",
        "initial_correct": initial,
        "final_correct": final,
        "initial_format_failure": False,
        "final_format_failure": False,
        "initial_status_correct": True,
        "final_status_correct": True,
        "verdict": verdict if verified else None,
        "second_verdict": None,
        "verifier_calls": 1 if verified else 0,
        "status": status or ("accepted" if verified else "unchecked"),
        "error": None,
        "tokens_in": 10,
        "tokens_out": 10,
        "cost_usd": 0.01 if condition == "none" else 0.02,
        "latency_s": 1.0,
        "n_calls": 1,
        "n_failed_calls": 0,
    }


def _rows(dedicated_runs=2):
    rows = []
    for task in TASKS:
        for run in (1, 2):
            right = not (task == "t2" and run == 1)  # one wrong draft
            rows.append(_row(task, "none", run, right, right))
            rows.append(_row(task, "self", run, right, right, verdict="accept"))
            if run <= dedicated_runs:
                if right:
                    rows.append(_row(task, "dedicated", run, True, True, verdict="accept"))
                else:  # caught and fixed
                    rows.append(_row(task, "dedicated", run, False, True, status="revised", verdict="reject"))
    return rows


@pytest.fixture
def results(tmp_path):
    def write(rows, log=None):
        out = tmp_path / "results"
        out.mkdir(exist_ok=True)
        (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (log or [])))
        return out

    return write


def test_progress_counts_saved_failed_and_complete(results):
    rows = _rows(dedicated_runs=1)
    rows.append(_row("t1", "self", 3, True, True, verdict="accept"))  # outside 2 repeats: ignored
    rows[1]["status"] = "failed"  # self t1 r1
    runs, _ = data.load_runs(results(rows))
    frame = data.progress(runs, TASKS, repeats=2).set_index("condition")
    assert frame.loc["none", ["runs_done", "runs_expected", "runs_failed"]].tolist() == [4, 4, 0]
    assert frame.loc["self", ["runs_done", "runs_failed"]].tolist() == [4, 1]
    assert frame.loc["dedicated", "runs_done"] == 2
    assert data.complete_conditions(frame.reset_index()) == ["none"]  # self has a failure, dedicated is short


def test_progress_counts_seeded_reviews_per_condition(results):
    runs, _ = data.load_runs(results(_rows()))
    seeded = [{"condition": "peer", "status": "reviewed"}, {"condition": "peer", "status": "failed"}]
    frame = data.progress(runs, TASKS, repeats=2, seeded_records=seeded, n_answers=31).set_index("condition")
    assert frame.loc["peer", ["seeded_done", "seeded_expected", "seeded_failed"]].tolist() == [2, 31, 1]


def test_headline_matches_the_report_and_keeps_study_order(results):
    runs, _ = data.load_runs(results(_rows()))
    report = data.headline(runs, ["dedicated", "none", "self"])
    summary = report["summary"].set_index("condition")
    assert list(report["summary"]["condition"]) == ["none", "self", "dedicated"]
    assert summary.loc["none", "task_success"] == pytest.approx(0.75)
    assert summary.loc["dedicated", "task_success"] == pytest.approx(1.0)
    assert summary.loc["dedicated", "fixed"] == 1
    with pytest.raises(ValueError):
        data.headline(runs, ["peer"])


def test_task_matrix_shows_right_out_of_saved_and_skips_failed(results):
    rows = _rows()
    rows.append({**_row("t2", "self", 3, False, False), "status": "failed"})
    runs, _ = data.load_runs(results(rows))
    matrix = data.task_matrix(runs, ["none", "self", "dedicated"]).set_index("task_id")
    assert matrix.loc["t2", ["none", "self", "dedicated"]].tolist() == ["1/2", "1/2", "2/2"]
    assert matrix.loc["t2", "best_gain"] == 1 and matrix.loc["t1", "best_gain"] == 0


def test_runs_for_task_is_in_study_order(results):
    runs, _ = data.load_runs(results(_rows()))
    table = data.runs_for_task(runs, "t2")
    assert list(table["condition"]) == ["none", "none", "self", "self", "dedicated", "dedicated"]


def test_answer_record_and_failed_checks(results):
    out = results(_rows())
    (out / "answers").mkdir()
    record = {"final_check": {"correct": False, "checks": [
        {"check_id": "a", "check_type": "STATUS", "passed": True},
        {"check_id": "b", "check_type": "REQUIRED_METRICS", "passed": False, "observed": {"x": 1}, "details": {"reason": "missing_observation"}},
    ]}}
    (out / "answers" / "self-m-t1-r1.json").write_text(json.dumps(record))
    assert data.answer_record("missing", out) is None
    loaded = data.answer_record("self-m-t1-r1", out)
    assert [c["check"] for c in data.failed_checks(loaded["final_check"])] == ["b"]
    assert data.failed_checks(None) == []


def test_spend_reads_the_run_log(results):
    log = [
        {"timestamp": "2026-11-20T01:00:00+00:00", "model": "m", "condition": "self", "cost_usd": 0.5},
        {"timestamp": "2026-11-21T01:00:00+00:00", "model": "m", "condition": "self", "cost_usd": 0.25, "error": "x"},
    ]
    summary = data.spend(results(_rows(), log))
    assert summary["cost_usd"] == pytest.approx(0.75)
    assert summary["failed"] == 1
    assert [d["name"] for d in summary["by_day"]] == ["2026-11-20", "2026-11-21"]


def test_seeded_tables_empty_without_records():
    metrics, audit = data.seeded_tables([])
    assert metrics.empty and audit.empty


def test_app_runs_on_a_small_results_folder(results, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    out = results(_rows())
    monkeypatch.setenv("ADVISORY_RESULTS_DIR", str(out))
    app = AppTest.from_file(str(data.ROOT / "scripts" / "dashboard.py"), default_timeout=60).run()
    assert not app.exception
    # dedicated is short of 3 repeats here, so with "only finished conditions" nothing is complete
    # (the study uses 3 repeats); include everything and check the headline metrics.
    app.sidebar.checkbox[0].uncheck().run()
    assert not app.exception
    shown = {m.label: m.value for m in app.metric}
    assert shown["none"] == "75.0%" and shown["dedicated"] == "100.0%"
    app.selectbox[0].select("t2").run()
    assert not app.exception
