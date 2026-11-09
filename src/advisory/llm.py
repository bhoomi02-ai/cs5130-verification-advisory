"""Single wrapper for every model call.

All agents must call models through `call_model` so that every call is logged
to results/runs.jsonl with the same fields. Do not call provider SDKs directly.
"""

from __future__ import annotations

import json
import os
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
    model: str  # exact model name, never an alias
    tokens_in: int
    tokens_out: int
    latency_s: float
    cost_usd: float
    raw_output: str
    temperature: float | None = None  # None means the provider default
    top_p: float | None = None  # None means the provider default
    thinking_level: str | None = None  # None means the provider default


# USD per 1M tokens (input, output), paid tier, standard (non-batch), text.
# Source: https://aistudio.google.com/docs/pricing, checked 2026-11-07.
# Output price covers thinking tokens. Add a model here before using it, so
# cost is never silently logged as zero by accident.
# Gemini 2.5 Flash / Flash-Lite are closed to new API users, so not listed.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.6-flash": (0.75, 3.75),  # rises to (1.50, 7.50) on 2027-01-01
    "gemini-3.8-flash": (0.75, 3.75),  # rises to (1.50, 7.50) on 2027-01-01
    # Gemma 4 is free tier only (no paid tier), so its cost really is zero.
    # Used for the cross-family verifier ablation.
    "gemma-4-26b-a4b-it": (0.0, 0.0),
    "gemma-4-31b-it": (0.0, 0.0),
}

_MAX_RETRIES = 5
_RETRY_STATUS = (429, 500, 503)

# How much the model reasons before answering (Gemini 3 thinking_level).
THINKING_LEVELS = ("minimal", "low", "medium", "high")


def compute_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    if model not in PRICING:
        raise ValueError(f"No pricing for model {model!r}; add it to PRICING")
    price_in, price_out = PRICING[model]
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def to_gemini(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split OpenAI-style messages into (system_instruction, contents).

    Each message is {"role": "system" | "user" | "assistant", "content": str}.
    Gemini calls the assistant role "model" and takes system text separately.
    """
    system_parts: list[str] = []
    contents: list[dict] = []
    for m in messages:
        role, text = m["role"], m["content"]
        if role == "system":
            system_parts.append(text)
        elif role in ("user", "assistant"):
            contents.append(
                {"role": "user" if role == "user" else "model", "parts": [{"text": text}]}
            )
        else:
            raise ValueError(f"Unknown message role {role!r}")
    if not contents:
        raise ValueError("messages must contain at least one user message")
    return ("\n\n".join(system_parts) or None), contents


_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from dotenv import load_dotenv
        from google import genai

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set; add it to .env")
        _gemini_client = genai.Client(api_key=key)
    return _gemini_client


def gemini_config(
    system: str | None,
    temperature: float | None,
    thinking_level: str | None,
    top_p: float | None = None,
) -> dict | None:
    """Build the Gemini request config; settings left as None use the model default."""
    config: dict = {}
    if system:
        config["system_instruction"] = system
    if temperature is not None:
        if not 0 <= temperature <= 2:
            raise ValueError(f"temperature must be between 0 and 2, got {temperature}")
        config["temperature"] = temperature
    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        config["top_p"] = top_p
    if thinking_level is not None:
        if thinking_level not in THINKING_LEVELS:
            raise ValueError(f"thinking_level must be one of {THINKING_LEVELS}, got {thinking_level!r}")
        config["thinking_config"] = {"thinking_level": thinking_level.upper()}
    return config or None


def _is_daily_quota(error) -> bool:
    """A per-day quota will not reset for hours, so retrying is pointless."""
    return error.code == 429 and "PerDay" in json.dumps(getattr(error, "details", None), default=str)


def _gemini_call(
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    thinking_level: str | None = None,
    top_p: float | None = None,
) -> tuple[str, int, int, float]:
    from google.genai import errors

    system, contents = to_gemini(messages)
    config = gemini_config(system, temperature, thinking_level, top_p)
    client = _get_gemini_client()
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config
            )
            break
        except errors.APIError as e:
            if e.code not in _RETRY_STATUS or attempt == _MAX_RETRIES - 1 or _is_daily_quota(e):
                raise
            time.sleep(2**attempt * 5)  # 5, 10, 20, 40 s; free tier is rate limited
    usage = response.usage_metadata
    tokens_in = usage.prompt_token_count or 0
    # Thinking tokens are billed as output, so count them as output.
    tokens_out = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    return response.text or "", tokens_in, tokens_out, compute_cost(model, tokens_in, tokens_out)


def _provider_call(
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    thinking_level: str | None = None,
    top_p: float | None = None,
) -> tuple[str, int, int, float]:
    """Return (text, tokens_in, tokens_out, cost_usd), dispatching on `model`."""
    if model.startswith(("gemini-", "gemma-")):
        return _gemini_call(model, messages, temperature, thinking_level, top_p)
    raise ValueError(f"No provider configured for model {model!r}")


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
    temperature: float | None = None,
    thinking_level: str | None = None,
    top_p: float | None = None,
) -> str:
    """Call `model` with `messages`, append a record to the run log, return the text.

    `temperature`, `top_p` and `thinking_level` (one of THINKING_LEVELS) are
    logged with the run; None leaves the provider default in place.
    """
    start = time.perf_counter()
    text, tokens_in, tokens_out, cost = _provider_call(
        model, messages, temperature, thinking_level, top_p
    )
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
            temperature=temperature,
            top_p=top_p,
            thinking_level=thinking_level,
        )
    )
    return text
