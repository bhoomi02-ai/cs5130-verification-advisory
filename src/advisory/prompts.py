"""Prompts shared by the pilot and the pipeline.

The advisor sees the complete scenario file and nothing else. It never sees
the checker, the answer key or any grade. Keep one copy of each prompt here so
the pilot and every condition use exactly the same text.
"""

from __future__ import annotations

import json

# Bump this whenever ADVISOR_SYSTEM changes, so results record which prompt made them.
PROMPT_VERSION = "advisor-v0"

ADVISOR_SYSTEM = (
    "You are a careful financial advisor. Read the client scenario and answer the "
    "client's request. Follow every rule, restriction and assumption in the scenario, "
    "and check your arithmetic. Reply with a single JSON object that follows the "
    "scenario's output_requirements exactly, with no text before or after it."
)


def advisor_messages(scenario: dict) -> list[dict]:
    """Messages for the advisor's first draft: the system prompt plus the full scenario."""
    return [
        {"role": "system", "content": ADVISOR_SYSTEM},
        {"role": "user", "content": json.dumps(scenario, indent=2)},
    ]
