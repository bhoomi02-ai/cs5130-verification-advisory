"""Single wrapper for every model call.

All agents must call models through `call_model` so that every call is logged
to results/runs.jsonl with the same fields. Do not call provider SDKs directly.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

RUN_LOG = Path(__file__).resolve().parents[2] / "results" / "runs.jsonl"


@dataclass
class RunRecord:
    timestamp: str
    task_id: str
    condition: str
    agent: str
    model: str  # pinned model string, e.g. a dated snapshot, never an alias
    tokens_in: int
    tokens_out: int
    latency_s: float
    cost_usd: float
    raw_output: str


def _provider_call(model: str, messages: list[dict]) -> tuple[str, int, int, float]:
    """Return (text, tokens_in, tokens_out, cost_usd).

    TODO: dispatch to the provider SDK based on `model`, read the API key from
    the environment (.env), and compute cost from the provider's token pricing.
    """
    raise NotImplementedError("Provider call not implemented yet")


def log_record(record: RunRecord, path: Path = RUN_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def call_model(
    *,
    task_id: str,
    condition: str,
    agent: str,
    model: str,
    messages: list[dict],
) -> str:
    """Call `model` with `messages`, append a record to the run log, return the text."""
    start = time.perf_counter()
    text, tokens_in, tokens_out, cost = _provider_call(model, messages)
    latency = time.perf_counter() - start
    log_record(
        RunRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_id=task_id,
            condition=condition,
            agent=agent,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=latency,
            cost_usd=cost,
            raw_output=text,
        )
    )
    return text
