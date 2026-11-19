"""Verifier reviews of the seeded answers in data/injected/.

Every checked condition's verifier (self, peer, dedicated, weakened,
cross_family) reviews each seeded answer once per run number. The seeded set
has 17 answers with one known error and 14 correct controls. This measures
only whether the verifier catches the error, so there is no revision and no
second review.

The review is the same function the main runs use (conditions.REVIEWS), so a
record's `reviews` list has the same fields as results/answers/<run_id>.json:
stage, verdict, issues, reason, error, raw, trailing, calls (plus calculations
and messages for dedicated, weakened and cross_family). There is always one
review, with stage "first". Join to data/evaluation/seeded_error_labels.csv
on `answer_id`.

Records are saved one per file in results/seeded/<run_id>.json. The run_id
holds the answer_id (seeded_run_id), and each logged call records the
answer_id as its task_id. That keeps seeded calls out of the main runs' files,
their usage totals in the run log, and anything that groups calls by task.

The verifier is shown only the scenario and the answer, never the seeded
file's description, error type or expected check.

Errors are handled the same way as pipeline.run_one: an API error is saved as
a record with status "failed"; a daily limit or any other exception is raised,
and nothing is saved for that run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from google.genai import errors as genai_errors

from advisory import llm
from advisory.conditions import REVIEWS, VERIFIED, VERIFIER_PROMPTS, verifier_settings
from advisory.drafts import Settings
from advisory.pipeline import SCENARIO_DIR, _daily_quota, _log_records, _logged_at, _totals, load_scenario

ROOT = Path(__file__).resolve().parents[2]
SEEDED_DIR = ROOT / "data" / "injected"
SEEDED_RESULTS_DIR = ROOT / "results" / "seeded"

# Settings a saved record must share with a new run before it is skipped as done.
_SETTING_FIELDS = (
    "model",
    "temperature",
    "top_p",
    "thinking_level",
    "verifier_model",
    "verifier_thinking_level",
    "verifier_prompt_version",
)


def seeded_run_id(condition: str, model: str, answer_id: str, run_number: int) -> str:
    """e.g. seeded-dedicated-gemini-3.6-flash-S01_policy_violation_sell_aapl-r1.

    conditions.make_run_id would give every seeded answer of a scenario the
    same ID, and the same one as the main run on that scenario.
    """
    return f"seeded-{condition}-{model}-{answer_id}-r{run_number}"


def all_answer_ids(seeded_dir: Path = SEEDED_DIR) -> list[str]:
    return sorted(p.stem for p in seeded_dir.glob("*.json"))


def load_seeded(answer_id: str, seeded_dir: Path = SEEDED_DIR) -> dict:
    return json.loads((seeded_dir / f"{answer_id}.json").read_text(encoding="utf-8"))


def answer_text(entry: dict) -> str:
    """The answer as the verifier sees it: the seeded response as JSON text, like an advisor's reply."""
    response = entry["response"]
    return response if isinstance(response, str) else json.dumps(response, indent=2)


def record_path(run_id: str, out_dir: Path | None = None) -> Path:
    return (out_dir or SEEDED_RESULTS_DIR) / f"{run_id}.json"


def settings_fields(condition: str, settings: Settings) -> dict:
    verifier = verifier_settings(condition, settings)
    return {
        "model": settings.model,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "thinking_level": settings.thinking_level,
        "verifier_model": verifier.model,
        "verifier_thinking_level": verifier.thinking_level,
        "verifier_prompt_version": VERIFIER_PROMPTS[condition],
    }


def saved_record(run_id: str, out_dir: Path | None = None) -> dict | None:
    path = record_path(run_id, out_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def check_same_settings(record: dict, condition: str, settings: Settings) -> None:
    """Refuse to mix a saved record made with other settings or another verifier prompt."""
    wanted = settings_fields(condition, settings)
    made = {k: record.get(k) for k in _SETTING_FIELDS}
    if made != wanted:
        raise ValueError(
            f"{record['run_id']} was made with other settings {made}, not {wanted}. "
            "Use the same settings, or move results/seeded/ aside to start a new set."
        )


def run_seeded_one(
    condition: str,
    answer_id: str,
    run_number: int,
    settings: Settings,
    *,
    seeded_dir: Path = SEEDED_DIR,
    scenario_dir: Path = SCENARIO_DIR,
    out_dir: Path | None = None,
    log_path: Path | None = None,
) -> dict:
    """Have `condition`'s verifier review one seeded answer once, save the record and return it."""
    if condition not in VERIFIED:
        raise ValueError(f"{condition!r} has no verifier; choose one of {', '.join(VERIFIED)}")
    started = datetime.now(timezone.utc)
    entry = load_seeded(answer_id, seeded_dir)
    scenario = load_scenario(entry["scenario_id"], scenario_dir)
    run_id = seeded_run_id(condition, settings.model, answer_id, run_number)
    verifier = verifier_settings(condition, settings)
    answer = answer_text(entry)

    def verifier_call(messages: list[dict]) -> str:
        return llm.call_model(
            task_id=answer_id,
            condition=condition,
            agent="verifier",
            model=verifier.model,
            messages=messages,
            temperature=verifier.temperature,
            top_p=verifier.top_p,
            thinking_level=verifier.thinking_level,
            run_id=run_id,
        )

    record = {
        "run_id": run_id,
        "answer_id": answer_id,
        "scenario_id": entry["scenario_id"],
        "error_type": entry["error_type"],
        "seeded_error": entry["error_type"] != "none",
        "condition": condition,
        "run_number": run_number,
        **settings_fields(condition, settings),
        "answer": answer,
    }
    try:
        review = REVIEWS[condition]("first", scenario, answer, verifier_call)
    except (genai_errors.APIError, llm.ProviderError) as e:
        if _daily_quota(e):  # not done yet, not failed: rerun tomorrow
            raise
        record.update(verdict=None, status="failed", verifier_calls=None, reviews=[], error=f"{type(e).__name__}: {e}"[:500])
    else:
        record.update(verdict=review["verdict"], status="reviewed", verifier_calls=review["calls"], reviews=[review], error=None)

    since = [r for r in _log_records({run_id}, log_path) if (_logged_at(r) or started) >= started]
    record.update(_totals(since))
    path = record_path(run_id, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record
