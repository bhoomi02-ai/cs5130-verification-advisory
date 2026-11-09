"""Run one condition on one scenario and save one result row.

Each row grades both the first draft (initial_correct) and the final answer
(final_correct) with the shared checker, and adds up the tokens, cost and time
of every model call in the run, including the shared first draft. The rows in
results/results.jsonl are the input to metrics.summarize().
"""

from __future__ import annotations

import json
from pathlib import Path

from advisory import llm
from advisory.checker import check
from advisory.conditions import CONDITIONS, make_run_id
from advisory.drafts import DRAFT_DIR, Settings, get_draft

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = ROOT / "data" / "scenarios"
RESULTS_PATH = ROOT / "results" / "results.jsonl"


def load_scenario(task_id: str, root: Path = SCENARIO_DIR) -> dict:
    return json.loads((root / f"{task_id}.json").read_text(encoding="utf-8"))


def all_task_ids(root: Path = SCENARIO_DIR) -> list[str]:
    return sorted(p.stem for p in root.glob("*.json"))


def usage_for(run_ids: list[str], log_path: Path | None = None) -> dict:
    """Add up tokens, cost and time of every logged call whose run_id is in run_ids."""
    log_path = log_path or llm.RUN_LOG
    totals = {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "latency_s": 0.0, "n_calls": 0, "n_failed_calls": 0}
    if not log_path.exists():
        return totals
    wanted = set(run_ids)
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("run_id") not in wanted:
                continue
            totals["tokens_in"] += rec.get("tokens_in", 0)
            totals["tokens_out"] += rec.get("tokens_out", 0)
            totals["cost_usd"] += rec.get("cost_usd", 0.0)
            totals["latency_s"] += rec.get("latency_s", 0.0)
            totals["n_calls"] += 1
            totals["n_failed_calls"] += 1 if rec.get("error") else 0
    return totals


def done_keys(results_path: Path = RESULTS_PATH) -> set[tuple]:
    """(task_id, condition, run_number, model) of every run already saved, so reruns skip them."""
    if not results_path.exists():
        return set()
    keys = set()
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                keys.add((row["task_id"], row["condition"], row["run_number"], row["model"]))
    return keys


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
    """Run `condition` once on `task_id`, grade it, append the row to results_path and return it."""
    scenario = load_scenario(task_id, scenario_dir)
    draft = get_draft(scenario, run_number, settings, root=draft_dir)
    run_id = make_run_id(condition, task_id, run_number, settings.model)
    result = CONDITIONS[condition](scenario, draft, run_id=run_id, settings=settings)

    first = check(task_id, result.draft)
    final = first if result.final == result.draft else check(task_id, result.final)

    row = {
        "run_id": run_id,
        "draft_run_id": draft.run_id,
        "task_id": task_id,
        "condition": condition,
        "run_number": run_number,
        "model": settings.model,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "thinking_level": settings.thinking_level,
        "prompt_version": draft.prompt_version,
        "initial_correct": bool(first["correct"]),
        "final_correct": bool(final["correct"]),
        "initial_format_failure": bool(first.get("format_failure", False)),
        "final_format_failure": bool(final.get("format_failure", False)),
        "verdict": result.verdict,
        "status": result.status,
        **usage_for([draft.run_id, run_id], log_path),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row
