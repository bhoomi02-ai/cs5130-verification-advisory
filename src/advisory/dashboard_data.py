"""Data for the Streamlit dashboard (scripts/dashboard.py).

Everything here only reads saved files under a results folder; nothing calls a
model. The numbers come from the same code as the report (report.build_report,
right_reason, spend), so the dashboard and scripts/analyze.py always agree.

The functions take the results folder as an argument, so the dashboard can
show the results of a backup (for example a checkout of study-results) as
well as the local ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from advisory import right_reason
from advisory.conditions import CONDITIONS
from advisory.pipeline import STUDY_TASKS_PATH, read_task_file
from advisory.report import build_report, load_results
from advisory.seeded import all_answer_ids
from advisory.spend import load_log, spend_summary

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
STUDY_REPEATS = 3
CONDITION_ORDER = ("none", "self", "peer", "dedicated", "weakened", "cross_family")
VERIFIED_ORDER = CONDITION_ORDER[1:]


def ordered(conditions) -> list[str]:
    """The given conditions in study order; unknown names go last."""
    known = [c for c in CONDITION_ORDER if c in set(conditions)]
    return known + sorted(set(conditions) - set(known))


def load_runs(results_dir: Path = RESULTS_DIR) -> tuple[pd.DataFrame, list[dict]]:
    """Rows of results.jsonl as a DataFrame, plus the rows the report rejects."""
    accepted, rejected = load_results(Path(results_dir) / "results.jsonl")
    return pd.DataFrame(accepted), rejected


def progress(
    runs: pd.DataFrame,
    tasks: list[str],
    repeats: int = STUDY_REPEATS,
    seeded_records: list[dict] | None = None,
    n_answers: int | None = None,
) -> pd.DataFrame:
    """Runs and seeded reviews saved per condition, against what the study needs.

    `done` counts saved rows of study tasks (failed ones included, shown
    separately), so it matches the coverage of scripts/run_condition.py.
    """
    expected = len(tasks) * repeats
    in_study = runs[runs["task_id"].isin(tasks) & runs["run_number"].between(1, repeats)] if len(runs) else runs
    rows = []
    for condition in CONDITION_ORDER:
        mine = in_study[in_study["condition"] == condition] if len(in_study) else in_study
        failed = int((mine["status"] == "failed").sum()) if len(mine) else 0
        row = {"condition": condition, "runs_done": len(mine), "runs_expected": expected, "runs_failed": failed}
        if condition != "none" and n_answers is not None:
            reviews = [r for r in (seeded_records or []) if r.get("condition") == condition]
            row["seeded_done"] = len(reviews)
            row["seeded_expected"] = n_answers
            row["seeded_failed"] = sum(1 for r in reviews if r.get("status") == "failed")
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["complete"] = (frame["runs_done"] >= frame["runs_expected"]) & (frame["runs_failed"] == 0)
    return frame


def complete_conditions(progress_frame: pd.DataFrame) -> list[str]:
    """Conditions with every study run saved and none failed."""
    return list(progress_frame.loc[progress_frame["complete"], "condition"])


def headline(runs: pd.DataFrame, conditions: list[str] | None = None, seed: int = 0) -> dict:
    """The report's summary, outcome and paired-comparison tables, for the chosen conditions."""
    if conditions is not None:
        runs = runs[runs["condition"].isin(conditions)]
    if runs.empty:
        raise ValueError("no runs for the chosen conditions")
    report = build_report(runs.to_dict("records"), seed=seed)
    for key in ("summary", "outcomes"):
        frame = report[key]
        report[key] = frame.set_index("condition").loc[ordered(frame["condition"])].reset_index()
    return report


def task_matrix(runs: pd.DataFrame, conditions: list[str] | None = None) -> pd.DataFrame:
    """One row per task: right finals out of saved runs for each condition, e.g. "2/3".

    Failed runs are not counted as saved. `none_right` and `best_gain` help
    sort the tasks where verification changed something.
    """
    usable = runs[runs["status"] != "failed"]
    conditions = ordered(conditions or usable["condition"].unique())
    rows = []
    for task_id, group in usable.groupby("task_id", sort=True):
        row = {"task_id": task_id}
        right = {}
        for condition in conditions:
            mine = group[group["condition"] == condition]
            right[condition] = int(mine["final_correct"].astype(bool).sum())
            row[condition] = f"{right[condition]}/{len(mine)}" if len(mine) else ""
        row["none_right"] = right.get("none", 0)
        row["best_gain"] = max((right[c] - right.get("none", 0) for c in conditions if c != "none"), default=0)
        row["worst_loss"] = min((right[c] - right.get("none", 0) for c in conditions if c != "none"), default=0)
        rows.append(row)
    return pd.DataFrame(rows)


def runs_for_task(runs: pd.DataFrame, task_id: str) -> pd.DataFrame:
    """Every run of one task, in study order of condition, then repeat."""
    mine = runs[runs["task_id"] == task_id].copy()
    mine["_order"] = mine["condition"].map({c: i for i, c in enumerate(CONDITION_ORDER)}).fillna(len(CONDITION_ORDER))
    columns = [
        "condition", "run_number", "initial_correct", "final_correct", "verdict", "second_verdict",
        "status", "verifier_calls", "cost_usd", "run_id",
    ]
    return mine.sort_values(["_order", "run_number"])[[c for c in columns if c in mine.columns]]


def answer_record(run_id: str, results_dir: Path = RESULTS_DIR) -> dict | None:
    """The saved draft, revision, reviews and checks of one run (results/answers/<run_id>.json)."""
    path = Path(results_dir) / "answers" / f"{run_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def failed_checks(check: dict | None) -> list[dict]:
    """The checks a graded answer failed, with what was observed, for display."""
    if not check:
        return []
    return [
        {"check": c.get("check_id"), "type": c.get("check_type"), "observed": c.get("observed"), "details": c.get("details")}
        for c in check.get("checks", [])
        if not c.get("passed")
    ]


def seeded_records(results_dir: Path = RESULTS_DIR) -> list[dict]:
    seeded_dir = Path(results_dir) / "seeded"
    return right_reason.load_seeded_records(seeded_dir) if seeded_dir.exists() else []


def seeded_tables(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Right-reason metrics per condition and the per-answer audit (empty frames if nothing saved)."""
    if not records:
        return pd.DataFrame(), pd.DataFrame()
    labels = right_reason.load_labels()
    metrics = right_reason.right_reason_metrics(records, labels)
    audit = right_reason.audit_table(records, labels)
    if not metrics.empty:
        metrics = metrics.set_index("condition").loc[ordered(metrics["condition"])].reset_index()
    return metrics, audit


def spend(results_dir: Path = RESULTS_DIR) -> dict:
    """Totals of the run log, overall and by model, condition and day."""
    return spend_summary(load_log(Path(results_dir) / "runs.jsonl"))


def study_tasks(path: Path = STUDY_TASKS_PATH) -> list[str]:
    return read_task_file(path)


def n_seeded_answers() -> int:
    return len(all_answer_ids())


assert set(CONDITION_ORDER) == set(CONDITIONS), "CONDITION_ORDER is out of date with conditions.CONDITIONS"
