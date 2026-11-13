"""Prompts shared by the pilot and the pipeline.

The advisor sees the complete scenario file and nothing else. It never sees
the checker, the answer key or any grade. Keep one copy of each prompt here so
the pilot and every condition use exactly the same text.
"""

from __future__ import annotations

import json

# Bump this whenever ADVISOR_SYSTEM changes, so results record which prompt made them.
# advisor-v1 is the prompt the model pilot (#1) measured the advisor with.
PROMPT_VERSION = "advisor-v1"

ADVISOR_SYSTEM = """\
You are a careful financial advisor working on a synthetic client scenario.
You receive the complete scenario as JSON. Answer the client_request exactly as
asked, respecting every restriction, cash flow and approved assumption in it.

Respond with a single JSON object and nothing else: no prose before or after,
no markdown code fences. The object must conform to
output_requirements.response_schema in the scenario, and follow every rule in
output_requirements (trades_rule, observations_rule, task_completion,
violations_rule, explanation_rule). Check your arithmetic before answering.
"""


def advisor_messages(scenario: dict) -> list[dict]:
    """Messages for the advisor's first draft: the system prompt plus the full scenario."""
    return [
        {"role": "system", "content": ADVISOR_SYSTEM},
        {"role": "user", "content": json.dumps(scenario, indent=2)},
    ]
