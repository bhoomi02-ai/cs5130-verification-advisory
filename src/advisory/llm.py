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
    run_id: str | None = None  # joins this call to the per-run result file
    error: str | None = None  # set when the call failed; tokens and cost are then 0


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
    # Groq, for the cross-family verifier (another company's model). Source:
    # https://console.groq.com/docs/model/openai/gpt-oss-20b, checked 2026-11-17.
    "openai/gpt-oss-20b": (0.075, 0.30),
}

# Models served by Groq's OpenAI-compatible API (exact Groq model IDs).
GROQ_MODELS = frozenset({"openai/gpt-oss-20b"})
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Groq's default output limit (1,024) can cut off a reasoning model mid-thought.
GROQ_MAX_COMPLETION_TOKENS = 8192
# Groq's reasoning_effort values for gpt-oss.
GROQ_REASONING_EFFORTS = ("low", "medium", "high")

_MAX_RETRIES = 5
_RETRY_STATUS = (429, 500, 503)

# Spending cap for the whole project, in USD (team decision, 2026-11-13).
# Override with ADVISORY_BUDGET_USD in .env, e.g. to give each teammate a share
# when several people spend from the same paid key on their own machines.
DEFAULT_BUDGET_USD = 20.0
# A call is refused unless this much budget is left, so the last call cannot
# overshoot the cap. One advisor call costs about $0.02; this leaves room for
# the longest calls.
_CALL_HEADROOM_USD = 0.10


class BudgetExceeded(RuntimeError):
    """Raised before a model call that could push total spend past the budget."""


def budget_usd() -> float:
    value = os.environ.get("ADVISORY_BUDGET_USD")
    return DEFAULT_BUDGET_USD if value in (None, "") else float(value)


def spent_usd(path: Path | None = None) -> float:
    """Total cost_usd of every call logged in the run log so far.

    Free-tier calls are logged at paid prices too, so this errs on the side of
    counting too much, never too little.
    """
    path = path or RUN_LOG
    if not path.exists():
        return 0.0
    total = 0.0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += json.loads(line).get("cost_usd") or 0.0
    return total


def check_budget(path: Path | None = None) -> None:
    """Raise BudgetExceeded if the next call could take spend past the budget."""
    budget, spent = budget_usd(), spent_usd(path)
    if spent + _CALL_HEADROOM_USD > budget:
        raise BudgetExceeded(
            f"Spent ${spent:.2f} of the ${budget:.2f} budget (results/runs.jsonl); "
            f"refusing a call that could go over. Raise ADVISORY_BUDGET_USD only if the team agrees."
        )



def _network_errors() -> tuple[type[BaseException], ...]:
    """Connection resets, timeouts and other transport failures of the SDK's HTTP client."""
    errors: list[type[BaseException]] = [ConnectionError, TimeoutError]
    for module in ("httpx2", "httpx"):  # google-genai uses one of these
        try:
            errors.append(__import__(module).TransportError)
        except (ImportError, AttributeError):
            pass
    return tuple(errors)


_NETWORK_ERRORS = _network_errors()

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

    if model not in PRICING:  # fail before any tokens are spent, not after
        raise ValueError(f"No pricing for model {model!r}; add it to PRICING")
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
        except _NETWORK_ERRORS:
            # A dropped connection or a timeout says nothing about the model, so
            # retry it like a 503. If the server had already answered, that call
            # may be billed twice, which costs cents, not a lost run.
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt * 5)
    text = response.text or ""
    tokens_in, tokens_out = usage_tokens(response.usage_metadata, text)
    return text, tokens_in, tokens_out, compute_cost(model, tokens_in, tokens_out)


# Used only when Gemini reports no output tokens for a reply that has text (seen
# live on short first replies of the dedicated verifier: total_token_count equal
# to the input, no candidates or thoughts count). The real count is unknown, so
# the estimate is deliberately high: about 3 characters per token for the text
# plus a thinking allowance above the most thinking seen on a verifier call
# (2,884). Overcounting keeps the $20 cap safe; undercounting would not.
ESTIMATED_THINKING_TOKENS = 3000


def usage_tokens(usage, text: str = "") -> tuple[int, int]:
    """(tokens_in, tokens_out) from Gemini's usage metadata, never undercounting output.

    Output is the answer plus thinking tokens, since both are billed as output,
    and at least total_token_count minus input. If Gemini reports no output at
    all for a non-empty reply, a high estimate is used instead; if it reports
    the answer but no thinking count, and the total shows no thinking either,
    the thinking allowance is added (see ESTIMATED_THINKING_TOKENS).
    """
    tokens_in = getattr(usage, "prompt_token_count", None) or 0
    tool_in = getattr(usage, "tool_use_prompt_token_count", None) or 0  # billed as input
    reported_out = (getattr(usage, "candidates_token_count", None) or 0) + (
        getattr(usage, "thoughts_token_count", None) or 0
    )
    total = getattr(usage, "total_token_count", None) or 0
    tokens_out = max(reported_out, total - tokens_in - tool_in)
    if tokens_out == 0 and text:
        tokens_out = -(-len(text) // 3) + ESTIMATED_THINKING_TOKENS
    elif text and getattr(usage, "thoughts_token_count", None) is None and total <= tokens_in + tool_in + reported_out:
        # The answer was counted but thinking was not reported, and the total
        # doesn't show any either: add the allowance rather than count no thinking.
        tokens_out += ESTIMATED_THINKING_TOKENS
    return tokens_in + tool_in, tokens_out


class ProviderError(RuntimeError):
    """A failed call to a provider without its own SDK error type (Groq). `code` is the HTTP status."""

    def __init__(self, code: int | None, message: str, *, daily_quota: bool = False):
        super().__init__(f"{code} {message}" if code else message)
        self.code = code
        self.daily_quota = daily_quota


def to_groq(messages: list[dict]) -> list[dict]:
    """OpenAI-style messages for Groq, with the system prompt moved into the first user turn.

    Groq advises against system prompts for its reasoning models ("include all
    instructions in the user message"), so the system text goes first in the
    first user message, unchanged. Every other turn is passed as is.
    """
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    out: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] not in ("user", "assistant"):
            raise ValueError(f"Unknown message role {m['role']!r}")
        out.append({"role": m["role"], "content": m["content"]})
    if not any(m["role"] == "user" for m in out):
        raise ValueError("messages must contain at least one user message")
    if system:
        first = next(i for i, m in enumerate(out) if m["role"] == "user")
        out[first] = {"role": "user", "content": system + "\n\n" + out[first]["content"]}
    return out


def groq_usage_tokens(usage: dict, content: str, reasoning: str) -> tuple[int, int]:
    """(tokens_in, tokens_out) from Groq's usage, never undercounting output.

    Groq's docs don't say whether reasoning is included in completion_tokens,
    so output is at least total minus input, and at least about a quarter of
    the characters of the answer plus the reasoning.
    """
    tokens_in = int(usage.get("prompt_tokens") or 0)
    reported = int(usage.get("completion_tokens") or 0)
    from_total = int(usage.get("total_tokens") or 0) - tokens_in
    from_text = -(-(len(content) + len(reasoning)) // 4)
    return tokens_in, max(reported, from_total, from_text)


def _groq_post(body: dict, key: str) -> dict:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(body).encode("utf-8"),
        # A generic User-Agent: Groq's CDN rejects urllib's default one (403, "error code: 1010").
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "python-client/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        full = e.read().decode("utf-8", "replace")
        detail = full[:500]
        remaining = e.headers.get("x-ratelimit-remaining-requests") if e.headers else None
        daily = e.code == 429 and (remaining == "0" or "per day" in detail.lower() or "(rpd)" in detail.lower() or "(tpd)" in detail.lower())
        error = ProviderError(e.code, detail, daily_quota=daily)
        error.retry_after = e.headers.get("retry-after") if e.headers else None
        error.body = full
        raise error from e
    except urllib.error.URLError as e:  # DNS, refused, reset: a network error, retried like one
        raise ConnectionError(str(e.reason)) from e


# Thinking allowance when Groq reports no usage (a refused tool call): deliberately high.
GROQ_ESTIMATED_REASONING_TOKENS = 3000


def _tool_call_reply(error: "ProviderError", body: dict) -> dict | None:
    """Recover the model's reply when Groq refuses it as an undeclared tool call.

    gpt-oss sometimes wraps a JSON reply (e.g. a calculator request) as a call
    to a tool we never declared; Groq then answers 400 tool_use_failed and puts
    the generation in failed_generation. Its "arguments" are the reply the
    model meant to give, so they become the reply text. Groq reports no usage
    for it, so tokens are estimated high: about a third of the characters
    sent and received, plus a reasoning allowance.
    """
    if error.code != 400 or "tool_use_failed" not in getattr(error, "body", ""):
        return None
    try:
        detail = json.loads(getattr(error, "body", ""))
        generation = json.loads(detail["error"]["failed_generation"])
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    arguments = generation.get("arguments") if isinstance(generation, dict) else None
    if not isinstance(arguments, dict):
        return None
    content = json.dumps(arguments)
    sent_chars = sum(len(m["content"]) for m in body["messages"])
    prompt = -(-sent_chars // 3)
    completion = -(-len(content) // 3) + GROQ_ESTIMATED_REASONING_TOKENS
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
    }


def _groq_call(
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    thinking_level: str | None = None,
    top_p: float | None = None,
) -> tuple[str, int, int, float]:
    if model not in PRICING:  # fail before any tokens are spent, not after
        raise ValueError(f"No pricing for model {model!r}; add it to PRICING")
    body: dict = {"model": model, "messages": to_groq(messages), "max_completion_tokens": GROQ_MAX_COMPLETION_TOKENS}
    if temperature is not None:
        if not 0 <= temperature <= 2:
            raise ValueError(f"temperature must be between 0 and 2, got {temperature}")
        body["temperature"] = temperature
    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        body["top_p"] = top_p
    if thinking_level is not None:
        if thinking_level not in GROQ_REASONING_EFFORTS:
            raise ValueError(f"thinking_level for Groq must be one of {GROQ_REASONING_EFFORTS}, got {thinking_level!r}")
        body["reasoning_effort"] = thinking_level

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set; add it to .env")

    for attempt in range(_MAX_RETRIES):
        try:
            data = _groq_post(body, key)
            break
        except ProviderError as e:
            recovered = _tool_call_reply(e, body)
            if recovered is not None:
                data = recovered
                break
            if e.code not in _RETRY_STATUS + (502,) or e.daily_quota or attempt == _MAX_RETRIES - 1:
                raise
            wait = 2**attempt * 5
            if getattr(e, "retry_after", None):
                try:
                    wait = max(wait, float(e.retry_after))
                except ValueError:
                    pass
            time.sleep(wait)
        except _NETWORK_ERRORS:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt * 5)
    message = (data.get("choices") or [{}])[0].get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or ""
    tokens_in, tokens_out = groq_usage_tokens(data.get("usage") or {}, content, reasoning)
    return content, tokens_in, tokens_out, compute_cost(model, tokens_in, tokens_out)


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
    if model in GROQ_MODELS:
        return _groq_call(model, messages, temperature, thinking_level, top_p)
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
    run_id: str | None = None,
) -> str:
    """Call `model` with `messages`, append a record to the run log, return the text.

    `temperature`, `top_p` and `thinking_level` (one of THINKING_LEVELS) are
    logged with the run; None leaves the provider default in place. `run_id`
    lets the log be joined to per-run results. A failed call is logged too,
    with `error` set, and then re-raised, so failures never silently vanish.

    Before calling, the total cost in the run log is checked against the
    budget (see check_budget); a call that could go over raises BudgetExceeded
    and is not made or logged.
    """
    if PRICING.get(model, (0.0, 0.0)) != (0.0, 0.0):  # free models (Gemma) cost nothing
        check_budget()
    start = time.perf_counter()
    common = dict(
        task_id=task_id,
        condition=condition,
        agent=agent,
        model=model,
        temperature=temperature,
        top_p=top_p,
        thinking_level=thinking_level,
        run_id=run_id,
    )
    try:
        text, tokens_in, tokens_out, cost = _provider_call(
            model, messages, temperature, thinking_level, top_p
        )
    except Exception as e:
        log_record(
            RunRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                tokens_in=0,
                tokens_out=0,
                latency_s=time.perf_counter() - start,
                cost_usd=0.0,
                raw_output="",
                error=f"{type(e).__name__}: {e}"[:500],
                **common,
            )
        )
        raise
    log_record(
        RunRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=time.perf_counter() - start,
            cost_usd=cost,
            raw_output=text,
            **common,
        )
    )
    return text
