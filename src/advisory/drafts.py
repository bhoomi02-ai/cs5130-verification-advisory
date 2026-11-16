"""First drafts: written once per (model, task, run number) and reused by every condition.

Every condition checks the exact same first draft, so any difference in results
comes from verification alone. Drafts are saved under results/drafts/<model>/.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from advisory import llm
from advisory.prompts import PROMPT_VERSION, advisor_messages

DRAFT_DIR = Path(__file__).resolve().parents[2] / "results" / "drafts"


@dataclass(frozen=True)
class Settings:
    """Generation settings for every model call in a run (study plan: temperature 0.7, top-p 0.9)."""

    model: str  # exact model name, never an alias
    temperature: float | None = 0.7
    top_p: float | None = 0.9
    thinking_level: str | None = None


@dataclass
class Draft:
    task_id: str
    run_number: int
    model: str
    text: str  # the advisor's raw answer
    run_id: str  # run_id of the advisor call, to find its tokens and cost in runs.jsonl
    prompt_version: str
    temperature: float | None
    top_p: float | None
    thinking_level: str | None


def draft_run_id(model: str, task_id: str, run_number: int) -> str:
    return f"draft-{model}-{task_id}-r{run_number}"


def draft_path(model: str, task_id: str, run_number: int, root: Path = DRAFT_DIR) -> Path:
    return root / model / f"{task_id}-r{run_number}.json"


def get_draft(scenario: dict, run_number: int, settings: Settings, *, root: Path = DRAFT_DIR) -> Draft:
    """Return the saved first draft, or ask the advisor once and save it."""
    task_id = scenario["scenario_id"]
    path = draft_path(settings.model, task_id, run_number, root)
    if path.exists():
        draft = Draft(**json.loads(path.read_text(encoding="utf-8")))
        made_with = (draft.temperature, draft.top_p, draft.thinking_level, draft.prompt_version)
        wanted = (settings.temperature, settings.top_p, settings.thinking_level, PROMPT_VERSION)
        if made_with != wanted:
            raise ValueError(
                f"Saved draft {path} was made with settings {made_with}, not {wanted}. "
                "Use a different draft folder instead of mixing settings."
            )
        return draft

    run_id = draft_run_id(settings.model, task_id, run_number)
    text = llm.call_model(
        task_id=task_id,
        condition="draft",
        agent="advisor",
        model=settings.model,
        messages=advisor_messages(scenario),
        temperature=settings.temperature,
        top_p=settings.top_p,
        thinking_level=settings.thinking_level,
        run_id=run_id,
    )
    draft = Draft(
        task_id=task_id,
        run_number=run_number,
        model=settings.model,
        text=text,
        run_id=run_id,
        prompt_version=PROMPT_VERSION,
        temperature=settings.temperature,
        top_p=settings.top_p,
        thinking_level=settings.thinking_level,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(draft), indent=2), encoding="utf-8")
    return draft
