"""Model pilot (issue #1): run the advisor alone on the scenarios, per model.

Goal from the proposal: pick the model whose baseline accuracy is near 50-80%,
so verification has errors to fix. This is the ``none`` condition run for
several candidate models.

Each run makes exactly one advisor call, and that first answer is the one
scored: there is no retry, so format failures are measured rather than hidden.
Every run gets a ``run_id`` that is written both to its result file and to the
run log (results/runs.jsonl), so results can be joined to tokens and cost.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from jsonschema import Draft202012Validator

from advisory.llm import call_model

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = ROOT / "data" / "scenarios"
PILOT_DIR = ROOT / "results" / "pilot"

# Bump the version whenever the prompt text changes, so results stay comparable.
PROMPT_VERSION = "advisor-v1"
ADVISOR_SYSTEM_PROMPT = """\
You are a careful financial advisor working on a synthetic client scenario.
You receive the complete scenario as JSON. Answer the client_request exactly as
asked, respecting every restriction, cash flow and approved assumption in it.

Respond with a single JSON object and nothing else: no prose before or after,
no markdown code fences. The object must conform to
output_requirements.response_schema in the scenario, and follow every rule in
output_requirements (trades_rule, observations_rule, task_completion,
violations_rule, explanation_rule). Check your arithmetic before answering.
"""

# Agreed study settings (README "Study settings"); thinking level is the model default.
DEFAULT_SETTINGS = {"temperature": 0.7, "top_p": 0.9, "thinking_level": None}

# A grader takes (scenario, answer) and returns either a bool or a checker
# result dict with "correct" and "checks" (see advisory.checker.check).
Grader = Callable[[dict, dict], "bool | dict"]


@dataclass
class PilotResult:
    run_id: str
    batch_id: str
    scenario_id: str
    model: str
    run_number: int
    prompt_version: str
    settings: dict
    schema_valid: bool
    errors: list[str]
    answer: dict | None
    raw_output: str
    api_error: str | None = None  # the call failed; there is no answer to score
    correct: bool | None = None  # None until graded, or when the call failed
    failures: list[str] = field(default_factory=list)  # "CHECK_TYPE:reason" labels


def checker_grade(scenario: dict, answer: dict) -> dict:
    """Grade with the hidden checker. Evaluation-side only: never pass to an agent."""
    from advisory.checker import check

    return check(scenario["scenario_id"], answer)


def failure_labels(result: dict) -> list[str]:
    """Summarize a checker result as "CHECK_TYPE:reason" labels, e.g. "REPLAY_TRADES:rule_violation"."""
    labels = []
    for c in result.get("checks", []):
        if not c.get("passed"):
            reason = (c.get("details") or {}).get("reason", "failed")
            labels.append(f"{c.get('check_type', c.get('check_id', '?'))}:{reason}")
    return labels


def load_scenarios(directory: Path = SCENARIO_DIR, ids: list[str] | None = None) -> list[dict]:
    paths = sorted(directory.glob("S*.json"))
    scenarios = [json.loads(p.read_text()) for p in paths]
    if ids:
        scenarios = [s for s in scenarios if s["scenario_id"] in ids]
    if not scenarios:
        raise FileNotFoundError(f"No scenarios found in {directory}")
    return scenarios


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_answer(text: str) -> dict:
    """Parse the model's JSON, tolerating a markdown code fence around it."""
    m = _FENCE.match(text)
    data = json.loads(m.group(1) if m else text)
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    return data


def validate_answer(scenario: dict, answer: dict) -> list[str]:
    """Return a list of problems; empty means the answer has the required shape."""
    schema = scenario["output_requirements"]["response_schema"]
    errors = [
        f"{'/'.join(map(str, e.absolute_path)) or '(root)'}: {e.message}"
        for e in Draft202012Validator(schema).iter_errors(answer)
    ]
    if answer.get("scenario_id") not in (None, scenario["scenario_id"]):
        errors.append(f"scenario_id is {answer['scenario_id']!r}, expected {scenario['scenario_id']!r}")
    if answer.get("answer_type") not in (None, scenario["answer_type"]):
        errors.append(f"answer_type is {answer['answer_type']!r}, expected {scenario['answer_type']!r}")
    return errors


def run_advisor(
    scenario: dict,
    model: str,
    run_number: int = 1,
    *,
    run_id: str,
    batch_id: str = "",
    settings: dict | None = None,
) -> PilotResult:
    """Ask the advisor once and keep that first answer as it is: no retry."""
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    sid = scenario["scenario_id"]
    messages = [
        {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(scenario, indent=2)},
    ]
    result = PilotResult(
        run_id=run_id,
        batch_id=batch_id,
        scenario_id=sid,
        model=model,
        run_number=run_number,
        prompt_version=PROMPT_VERSION,
        settings=settings,
        schema_valid=False,
        errors=[],
        answer=None,
        raw_output="",
    )
    try:
        text = call_model(
            task_id=sid,
            condition="pilot",
            agent="advisor",
            model=model,
            messages=messages,
            run_id=run_id,
            **settings,
        )
    except Exception as e:  # logged by call_model; recorded here so the run is not lost
        result.api_error = f"{type(e).__name__}: {e}"[:500]
        result.errors = ["API call failed"]
        return result
    result.raw_output = text
    try:
        result.answer = parse_answer(text)
        result.errors = validate_answer(scenario, result.answer)
    except (json.JSONDecodeError, ValueError) as e:
        result.answer, result.errors = None, [f"not valid JSON: {e}"]
    result.schema_valid = not result.errors
    return result


def planned_calls(models: list[str], scenarios: list[dict], repeats: int) -> int:
    return len(models) * len(scenarios) * repeats


def run_pilot(
    models: list[str],
    scenarios: list[dict],
    repeats: int = 1,
    grade: Grader | None = None,
    out_dir: Path = PILOT_DIR,
    settings: dict | None = None,
    max_calls: int | None = None,
    batch_id: str | None = None,
) -> list[PilotResult]:
    """Run every model on every scenario `repeats` times and save each result.

    Results go to out_dir/<batch_id>/<model>/<scenario>_run<n>.json, so a new
    batch never overwrites an earlier one. Refuses to start if the run would
    make more than `max_calls` model calls.
    """
    n = planned_calls(models, scenarios, repeats)
    if max_calls is not None and n > max_calls:
        raise ValueError(f"This pilot would make {n} model calls, more than max_calls={max_calls}")
    batch_id = batch_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    results = []
    for model in models:
        for scenario in scenarios:
            sid = scenario["scenario_id"]
            for run in range(1, repeats + 1):
                run_id = f"{batch_id}-{model}-{sid}-r{run}"
                r = run_advisor(scenario, model, run, run_id=run_id, batch_id=batch_id, settings=settings)
                if r.api_error:
                    r.failures = ["API:error"]  # no answer, so correct stays None
                elif not grade:
                    pass
                elif not r.schema_valid:
                    r.correct, r.failures = False, ["SCHEMA:format"]
                else:
                    graded = grade(scenario, r.answer)
                    if isinstance(graded, dict):
                        r.correct, r.failures = bool(graded["correct"]), failure_labels(graded)
                    else:
                        r.correct = bool(graded)
                path = out_dir / batch_id / model / f"{sid}_run{run}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise FileExistsError(f"{path} already exists; use a new batch_id")
                path.write_text(json.dumps(asdict(r), indent=2))
                results.append(r)
                status = "API ERROR" if r.api_error else ("valid" if r.schema_valid else "INVALID")
                print(
                    f"{model:24} {sid} run{run}: {status}"
                    + ("" if r.correct is None else f", {'correct' if r.correct else 'wrong'}")
                    + (f" ({', '.join(r.failures)})" if r.failures and not r.correct else "")
                )
    return results


def summarize(results: list[PilotResult]) -> list[dict]:
    """One row per model. API errors are counted, never silently dropped."""
    rows = []
    for model in dict.fromkeys(r.model for r in results):
        rs = [r for r in results if r.model == model]
        answered = [r for r in rs if not r.api_error]
        graded = [r for r in answered if r.correct is not None]
        rows.append(
            {
                "model": model,
                "runs": len(rs),
                "api_errors": len(rs) - len(answered),
                "schema_valid": (sum(r.schema_valid for r in answered) / len(answered)) if answered else None,
                "accuracy": (sum(r.correct for r in graded) / len(graded)) if graded else None,
                "graded_runs": len(graded),
                "failures": dict(Counter(f for r in rs for f in r.failures).most_common()),
            }
        )
    return rows
