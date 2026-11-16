"""The dedicated verifier's review loop: calculator rounds, then a verdict.

The verifier replies either with {"calculate": [...]} or with its verdict. Each
calculator request is evaluated with the sandboxed calculator and the results
are sent back, for at most prompts.MAX_CALCULATOR_ROUNDS rounds; after that it
is asked for its verdict once more. A reply that is neither a calculator request
nor a readable verdict ends the review with verdict None, so it is counted as a
missing verdict rather than guessed.

The model call is passed in as `call(messages) -> str`, so the pipeline decides
the model, settings and run_id (normally a call_model partial) and every round
is logged like any other call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from advisory.calculator import calculate
from advisory.prompts import (
    FINAL_VERDICT_REQUEST,
    MAX_CALCULATOR_ROUNDS,
    MAX_EXPRESSIONS_PER_ROUND,
    calculator_results_message,
    dedicated_review_messages,
)
from advisory.verdict import Verdict, parse_verdict

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@dataclass
class DedicatedReview:
    verdict: Verdict | None  # None when the verifier never gave a readable verdict
    messages: list[dict]  # the whole conversation, for debugging and the run record
    calculations: list[dict] = field(default_factory=list)  # every calculator call and its result
    n_calls: int = 0  # model calls made by this review
    error: str | None = None  # why verdict is None


def calculator_request(text: str) -> list[str] | None:
    """The expressions if `text` is a calculator request, else None."""
    match = _FENCE.match(text or "")
    try:
        data = json.loads(match.group(1) if match else text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or "calculate" not in data:
        return None
    expressions = data["calculate"]
    if isinstance(expressions, str):
        expressions = [expressions]
    if not isinstance(expressions, list):
        return None
    return [str(e) for e in expressions]


def dedicated_review(scenario: dict, answer: str, call: Callable[[list[dict]], str]) -> DedicatedReview:
    """Run the dedicated verifier on `answer`. Never raises for a bad reply; `call` errors propagate."""
    messages = dedicated_review_messages(scenario, answer)
    review = DedicatedReview(verdict=None, messages=messages)
    rounds = 0
    while True:
        reply = call(messages)
        review.n_calls += 1
        messages.append({"role": "assistant", "content": reply})

        expressions = calculator_request(reply)
        if expressions is None:
            try:
                review.verdict = parse_verdict(reply)
            except ValueError as e:
                review.error = f"unreadable reply: {e}"
            return review

        if rounds >= MAX_CALCULATOR_ROUNDS:  # asked for the verdict and got another request
            review.error = "no verdict after the last calculator round"
            return review
        rounds += 1
        results = [calculate(e) for e in expressions[:MAX_EXPRESSIONS_PER_ROUND]]
        results += [
            {"expression": e, "error": f"not evaluated: at most {MAX_EXPRESSIONS_PER_ROUND} expressions per request"}
            for e in expressions[MAX_EXPRESSIONS_PER_ROUND:]
        ]
        review.calculations.extend(results)
        result_turn = calculator_results_message(results)
        if rounds >= MAX_CALCULATOR_ROUNDS:
            result_turn["content"] += "\n\n" + FINAL_VERDICT_REQUEST
        messages.append(result_turn)
