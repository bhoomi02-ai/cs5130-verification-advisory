"""Model pilot (issue #1): run the advisor alone on the scenarios, per model.

Goal from the proposal: pick the model whose baseline accuracy is near 50-80%,
so verification has errors to fix. This is the ``none`` condition run for
several candidate models. Each answer is checked against the scenario's
response schema; correctness is graded by an optional ``grade`` function
(the hidden checker, plugged in when it is ready).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from jsonschema import Draft202012Validator

from advisory.llm import call_model

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = ROOT / "data" / "scenarios"
PILOT_DIR = ROOT / "results" / "pilot"

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

FIX_PROMPT = """\
Your previous response could not be accepted: {problem}
Respond again with only the corrected JSON object.
"""

# A grader takes (scenario, answer) and returns either a bool or a checker
# result dict with "correct" and "checks" (see advisory.checker.check).
Grader = Callable[[dict, dict], "bool | dict"]


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


@dataclass
class PilotResult:
    scenario_id: str
    model: str
    run_number: int
    attempts: int
    schema_valid: bool
    errors: list[str]
    answer: dict | None
    raw_outputs: list[str] = field(default_factory=list)
    correct: bool | None = None  # None until a grader is supplied
    failures: list[str] = field(default_factory=list)  # failed checks, from the checker


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


def run_advisor(scenario: dict, model: str, run_number: int = 1, max_attempts: int = 2) -> PilotResult:
    """Ask the advisor once; on unparseable or schema-invalid output, ask it to fix it."""
    sid = scenario["scenario_id"]
    messages = [
        {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(scenario, indent=2)},
    ]
    raw_outputs: list[str] = []
    errors: list[str] = []
    answer: dict | None = None
    for _ in range(max_attempts):
        text = call_model(task_id=sid, condition="pilot", agent="advisor", model=model, messages=messages)
        raw_outputs.append(text)
        try:
            answer = parse_answer(text)
            errors = validate_answer(scenario, answer)
        except (json.JSONDecodeError, ValueError) as e:
            answer, errors = None, [f"not valid JSON: {e}"]
        if not errors:
            break
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": FIX_PROMPT.format(problem="; ".join(errors[:5]))},
        ]
    return PilotResult(
        scenario_id=sid,
        model=model,
        run_number=run_number,
        attempts=len(raw_outputs),
        schema_valid=not errors,
        errors=errors,
        answer=answer,
        raw_outputs=raw_outputs,
    )


def run_pilot(
    models: list[str],
    scenarios: list[dict],
    repeats: int = 1,
    grade: Grader | None = None,
    out_dir: Path = PILOT_DIR,
) -> list[PilotResult]:
    """Run every model on every scenario `repeats` times and save each result."""
    results = []
    for model in models:
        for scenario in scenarios:
            for run in range(1, repeats + 1):
                r = run_advisor(scenario, model, run)
                if grade and r.schema_valid:
                    graded = grade(scenario, r.answer)
                    if isinstance(graded, dict):
                        r.correct = bool(graded["correct"])
                        r.failures = failure_labels(graded)
                    else:
                        r.correct = bool(graded)
                elif grade:
                    r.correct = False  # an unusable answer counts as wrong
                    r.failures = ["SCHEMA:format"]
                path = out_dir / model / f"{r.scenario_id}_run{run}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(asdict(r), indent=2))
                results.append(r)
                print(
                    f"{model:24} {r.scenario_id} run{run}: "
                    f"{'valid' if r.schema_valid else 'INVALID'} after {r.attempts} attempt(s)"
                    + ("" if r.correct is None else f", {'correct' if r.correct else 'wrong'}")
                    + (f" ({', '.join(r.failures)})" if r.failures else "")
                )
    return results


def summarize(results: list[PilotResult]) -> list[dict]:
    rows = []
    for model in dict.fromkeys(r.model for r in results):
        rs = [r for r in results if r.model == model]
        graded = [r for r in rs if r.correct is not None]
        rows.append(
            {
                "model": model,
                "runs": len(rs),
                "schema_valid": sum(r.schema_valid for r in rs) / len(rs),
                "valid_first_try": sum(r.schema_valid and r.attempts == 1 for r in rs) / len(rs),
                "accuracy": (sum(r.correct for r in graded) / len(graded)) if graded else None,
                "failures": dict(Counter(f for r in rs for f in r.failures).most_common()),
            }
        )
    return rows
