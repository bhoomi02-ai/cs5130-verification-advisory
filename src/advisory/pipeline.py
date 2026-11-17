"""Run one condition on one scenario and save one result row.

Each row grades both the first draft (initial_correct) and the final answer
(final_correct) with the shared checker, and adds up the tokens, cost and time
of the run's model calls, including the shared first draft. The rows in
results/results.jsonl are the input to metrics.summarize(). Each row also
records whether the checker's STATUS check passed (initial_status_correct,
final_status_correct), for status-only accuracy in the report. self and peer
rows also record the second verdict (on the revision) and the verifier and
revision prompt versions, and every graded run saves its answers (draft,
revision, final), the verifier replies and both check() results to
results/answers/<run_id>.json, so fixed and broken answers can be checked.

Usage is counted per attempt: a rerun is charged only for its own calls, plus
the call that made a first draft saved earlier, never for calls from an earlier
attempt that stopped on the daily quota.

A run whose model call fails with an API error (after llm.call_model's own
retries) is saved too, with status "failed" and the error, so failed runs are
counted per condition instead of disappearing. Two errors are not saved as
runs, because they say nothing about the condition: the daily quota (the run
is simply not done yet; run the same command again tomorrow) and errors in our
own code or setup (a bug to fix, not a failed run).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from google.genai import errors as genai_errors

from advisory import llm
from advisory.checker import check
from advisory.conditions import CONDITIONS, VERIFIED, make_run_id
from advisory.drafts import DRAFT_DIR, PROMPT_VERSION, Settings, draft_run_id, get_draft
from advisory.prompts import REVISION_PROMPT_VERSION, VERIFIER_PROMPT_VERSION

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = ROOT / "data" / "scenarios"
RESULTS_PATH = ROOT / "results" / "results.jsonl"


def load_scenario(task_id: str, root: Path = SCENARIO_DIR) -> dict:
    return json.loads((root / f"{task_id}.json").read_text(encoding="utf-8"))


def all_task_ids(root: Path = SCENARIO_DIR) -> list[str]:
    return sorted(p.stem for p in root.glob("*.json"))


def _log_records(run_ids: set[str], log_path: Path | None) -> list[dict]:
    """Logged calls whose run_id is in run_ids, in the order they were written."""
    log_path = log_path or llm.RUN_LOG
    if not log_path.exists():
        return []
    records = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("run_id") in run_ids:
                    records.append(rec)
    return records


def _totals(records: list[dict]) -> dict:
    totals = {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "latency_s": 0.0, "n_calls": 0, "n_failed_calls": 0}
    for rec in records:
        totals["tokens_in"] += rec.get("tokens_in", 0)
        totals["tokens_out"] += rec.get("tokens_out", 0)
        totals["cost_usd"] += rec.get("cost_usd", 0.0)
        totals["latency_s"] += rec.get("latency_s", 0.0)
        totals["n_calls"] += 1
        totals["n_failed_calls"] += 1 if rec.get("error") else 0
    return totals


def _logged_at(rec: dict) -> datetime | None:
    try:
        when = datetime.fromisoformat(rec["timestamp"])
    except (KeyError, TypeError, ValueError):
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def usage_for(run_ids: list[str], log_path: Path | None = None) -> dict:
    """Add up tokens, cost and time of every logged call whose run_id is in run_ids."""
    return _totals(_log_records(set(run_ids), log_path))


def attempt_usage(draft_id: str, run_id: str, started: datetime, log_path: Path | None = None) -> dict:
    """Tokens, cost and time of one attempt of a run.

    Counts the calls logged at or after `started` for the run and its first
    draft. If the first draft was saved by an earlier attempt, the call that
    made it (its latest successful record) is added. Calls from earlier
    attempts, like one that stopped on the daily quota, are left out.
    """
    records = _log_records({draft_id, run_id}, log_path)
    picked = [r for r in records if (_logged_at(r) or datetime.min.replace(tzinfo=timezone.utc)) >= started]
    if not any(r.get("run_id") == draft_id for r in picked):
        made = [r for r in records if r.get("run_id") == draft_id and not r.get("error")]
        if made:
            picked.append(made[-1])
    return _totals(picked)


def _rows(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    with results_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _key(row: dict) -> tuple:
    return (row["task_id"], row["condition"], row["run_number"], row["model"])


def done_keys(results_path: Path = RESULTS_PATH, settings: Settings | None = None) -> set[tuple]:
    """(task_id, condition, run_number, model) of every run already saved, so reruns skip them.

    Failed runs count as saved: they are results, and running them again would
    add a second row for the same run, which metrics.summarize() rejects
    (scripts/run_condition.py --retry-failed replaces them instead).

    With `settings`, rows for the same model made with other settings or another
    prompt version are refused, the same way get_draft refuses a saved draft.
    For self and peer rows the verifier and revision prompt versions must match too.
    """
    rows = _rows(results_path)
    if settings is not None:
        wanted = (settings.temperature, settings.top_p, settings.thinking_level, PROMPT_VERSION)
        wanted_review = (VERIFIER_PROMPT_VERSION, REVISION_PROMPT_VERSION)
        for row in rows:
            if row["model"] != settings.model:
                continue
            made = (row.get("temperature"), row.get("top_p"), row.get("thinking_level"), row.get("prompt_version"))
            if made != wanted:
                raise ValueError(
                    f"{results_path.name} already has {settings.model} runs made with other settings {made}, not {wanted}. "
                    "Use the same settings, or move the file aside to start a new one."
                )
            made_review = (row.get("verifier_prompt_version"), row.get("revision_prompt_version"))
            if row["condition"] in VERIFIED and made_review != wanted_review:
                raise ValueError(
                    f"{results_path.name} already has {row['condition']} runs made with other verifier prompts "
                    f"{made_review}, not {wanted_review}. Move the file aside to start a new one."
                )
    return {_key(row) for row in rows}


def failed_keys(results_path: Path = RESULTS_PATH) -> set[tuple]:
    """Keys of the runs saved with status "failed"."""
    return {_key(row) for row in _rows(results_path) if row.get("status") == "failed"}


def drop_failed(keys: set[tuple], results_path: Path = RESULTS_PATH) -> int:
    """Remove the failed rows whose key is in `keys`, so those runs can be run again. Returns how many."""
    rows = _rows(results_path)
    kept = [row for row in rows if not (row.get("status") == "failed" and _key(row) in keys)]
    if len(kept) == len(rows):
        return 0
    tmp = results_path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8")
    os.replace(tmp, results_path)
    return len(rows) - len(kept)


def status_correct(graded: dict) -> bool | None:
    """Whether check()'s STATUS check passed.

    A format failure has no readable status, so it counts as wrong, like
    initial_correct. None means the scenario has no STATUS check.
    """
    if graded.get("format_failure"):
        return False
    for row in graded.get("checks", []):
        if row["check_type"] == "STATUS":
            return bool(row["passed"])
    return None


def run_one(
    condition: str,
    task_id: str,
    run_number: int,
    settings: Settings,
    *,
    results_path: Path = RESULTS_PATH,
    draft_dir: Path = DRAFT_DIR,
    scenario_dir: Path = SCENARIO_DIR,
    log_path: Path | None = None,
) -> dict:
    """Run `condition` once on `task_id`, grade it, append the row to results_path and return it.

    An API error is saved as a row with status "failed". The daily quota and
    any other exception are raised, and nothing is saved for the run.
    """
    started = datetime.now(timezone.utc)
    scenario = load_scenario(task_id, scenario_dir)
    run_id = make_run_id(condition, task_id, run_number, settings.model)
    draft_id = draft_run_id(settings.model, task_id, run_number)
    base = {
        "run_id": run_id,
        "draft_run_id": draft_id,
        "task_id": task_id,
        "condition": condition,
        "run_number": run_number,
        "model": settings.model,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "thinking_level": settings.thinking_level,
        "verifier_prompt_version": VERIFIER_PROMPT_VERSION if condition in VERIFIED else None,
        "revision_prompt_version": REVISION_PROMPT_VERSION if condition in VERIFIED else None,
    }

    try:
        draft = get_draft(scenario, run_number, settings, root=draft_dir)
        result = CONDITIONS[condition](scenario, draft, run_id=run_id, settings=settings)
    except genai_errors.APIError as e:
        if llm._is_daily_quota(e):  # not done yet, not failed: rerun tomorrow
            raise
        row = {
            **base,
            "prompt_version": PROMPT_VERSION,
            # A failed run has no answer to grade. metrics.summarize() leaves
            # status "failed" out of every rate, so these only keep the
            # columns boolean.
            "initial_correct": False,
            "final_correct": False,
            "initial_format_failure": False,
            "final_format_failure": False,
            "initial_status_correct": False,
            "final_status_correct": False,
            "verdict": None,
            "second_verdict": None,
            "status": "failed",
            "error": f"{type(e).__name__}: {e}"[:500],
            **attempt_usage(draft_id, run_id, started, log_path),
        }
        _append(row, results_path)
        return row

    first = check(task_id, result.draft)
    final = first if result.final == result.draft else check(task_id, result.final)

    row = {
        **base,
        "prompt_version": draft.prompt_version,
        "initial_correct": bool(first["correct"]),
        "final_correct": bool(final["correct"]),
        "initial_format_failure": bool(first.get("format_failure", False)),
        "final_format_failure": bool(final.get("format_failure", False)),
        "initial_status_correct": status_correct(first),
        "final_status_correct": status_correct(final),
        "verdict": result.verdict,
        "second_verdict": result.second_verdict,
        "status": result.status,
        "error": None,
        **attempt_usage(draft.run_id, run_id, started, log_path),
    }
    _save_answers(result, first, final, results_path.parent / "answers")
    _append(row, results_path)
    return row


def _save_answers(result, first_check: dict, final_check: dict, answers_dir: Path) -> None:
    """Keep the run's answers, verifier replies and check() results, joined to the row by run_id."""
    record = {
        "run_id": result.run_id,
        "task_id": result.task_id,
        "condition": result.condition,
        "run_number": result.run_number,
        "draft": result.draft,
        "revision": result.revision,
        "final": result.final,
        "verdict": result.verdict,
        "second_verdict": result.second_verdict,
        "status": result.status,
        "reviews": result.reviews,
        "draft_check": first_check,
        "final_check": final_check,
    }
    answers_dir.mkdir(parents=True, exist_ok=True)
    (answers_dir / f"{result.run_id}.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def _append(row: dict, results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
