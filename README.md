# CS 5130 Verification Strategies for Multi-Agent Financial Advisory

Northeastern CS 5130, Fall 2026. Theme 6 (Agentic Systems), Track A (Verification).

## Research question

How do different verification strategies affect the correctness of a multi-agent
financial advisory system on synthetic client scenarios? We compare four
conditions, with everything else held fixed. Results are scored by a
deterministic constraint checker that is never shown to any agent.

## Conditions

| Condition   | Who verifies                     | Description                                               |
|-------------|----------------------------------|-----------------------------------------------------------|
| `none`      | Nobody                           | The advisor answers once; its output is final.            |
| `self`      | The advisor itself               | The advisor reviews and revises its own answer.           |
| `peer`      | Another advisor                  | A second advisor of the same role critiques; the first revises. |
| `dedicated` | A separate verifier agent        | A verifier with its own prompt checks the answer before it is final. |

## Team

- Bhoomika
- Faisal Alsabih
- Tejas Vanjari
- Dhanya Kini Bailoor

## Layout

```
src/advisory/
  llm.py          single wrapper for all model calls; logs to results/runs.jsonl
  schemas.py      Pydantic models: Trade, Claim, Recommendation
  conditions.py   the four conditions and the CONDITIONS registry
  checker.py      deterministic constraint checker (hidden from agents)
  generator.py    synthetic client scenario generator
tests/            pytest suite
data/scenarios/   generated scenarios
results/          run logs and metrics (runs.jsonl)
docs/             proposal, architecture diagram, notes
scripts/          entry points for generating, running, and analysing
```

## Setup

```bash
git clone https://github.com/bhoomi02-ai/cs5130-verification-advisory.git
cd cs5130-verification-advisory
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in your own keys; .env is gitignored
pytest
```

Requires Python 3.10 or newer. Never commit `.env` or any API key.

## Workflow

- One branch per task, named after the issue, e.g. `generator/scenario-schema`.
- Open a pull request into `main` and link the issue (`Closes #N`).
- At least one teammate reviews before merging. `pytest` must pass.
- Every model call goes through `advisory.llm.call_model` so it is logged.
- Pin exact model strings (dated snapshots), never aliases.
