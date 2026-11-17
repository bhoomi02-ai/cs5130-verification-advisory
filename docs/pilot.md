# Model pilot (issue #1)

**Goal (from the proposal):** pick one small, inexpensive model for every role whose baseline accuracy on the `none` condition is near **50–80%**, so verification has real errors to fix but the model isn't failing on everything.

**Result:** **`gemini-3.6-flash`**, at 83% (25/30). The thinking level is still to be settled (see [Decision](#decision)).

## Setup

| | |
|---|---|
| Condition | `none` (one advisor call, first answer scored as is, no retry) |
| Scenarios | S01–S10, 3 repeats each (T03-v001 was added after the pilot) |
| Prompt | `advisor-v1` (`src/advisory/prompts.py`), the same prompt the pipeline uses |
| Settings | temperature 0.7, top-p 0.9, thinking level = model default |
| Grading | the hidden checker (`advisory.checker.check`); one outer code fence is stripped before grading |
| Code | `src/advisory/pilot.py`, `scripts/run_pilot.py` (#20) |
| Batches | `results/pilot/20261113T021603Z` (3.6 Flash, finished with `--resume` after the free-tier quota), `results/pilot/20261113T022642Z` (3.5 Flash-Lite) |

## Results

| Model | Runs | Correct | Valid JSON | Mean cost / call | Median latency |
|---|---|---|---|---|---|
| **gemini-3.6-flash** | 30 | **83% (25/30)** | 100% | $0.018 | 27 s |
| gemini-3.5-flash-lite | 20 | 30% (6/20) | 100% | $0.003 | 3 s |
| gemini-3.1-flash-lite | 30 | 3% (1/30) | | $0.002 | 3 s |

gemini-3.1-flash-lite is from an earlier batch with the first pilot prompt (#20), so it isn't directly comparable. It's listed only to show how far below the target it was. Costs are at paid-tier prices; the pilot itself ran on the free tier. Mean tokens per call: 2,854 in for both models, and 4,156 out for 3.6 Flash (mostly thinking) versus 835 for 3.5 Flash-Lite.

**Correct runs per scenario**

| | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | S10 |
|---|---|---|---|---|---|---|---|---|---|---|
| gemini-3.6-flash | 3/3 | 3/3 | 3/3 | **0/3** | 3/3 | 2/3 | 2/3 | 3/3 | 3/3 | 3/3 |
| gemini-3.5-flash-lite | 0/2 | 0/2 | 2/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 2/2 | 2/2 |

**What went wrong**

- **gemini-3.6-flash:** only arithmetic slips: 4 × `INDEPENDENT_RESULTS:value_mismatch` (S04 ×3, S06 ×1) and 1 × `REPLAY_TRADES:value_mismatch` (S07). For example, on S04 it reported Fund B's five-year value as $145,470.34 or $145,469.95 instead of $145,469.35, against a $0.01 tolerance. These are the errors a calculator-equipped verifier should catch.
- **gemini-3.5-flash-lite:** mostly broken trade rules (10 × `REPLAY_TRADES:rule_violation`), plus arithmetic errors on S04 and S06. It only solved S03 and the two infeasible scenarios.
- **Format:** every answer from both models was valid JSON in the required shape. With `advisor-v1`, which says "no markdown code fences", only 1 of 30 gemini-3.6-flash answers was fenced.

## Decision

gemini-3.6-flash is the only candidate near the target. **83% is just above 80%**: with 10 scenarios the uncertainty is roughly ±10 points, so it's within noise of the range. Two caveats:

1. Its errors come from only **3 of 10 scenarios** (S04, S06, S07), so on its own the advisor gives the verifiers few distinct errors. The **seeded error set** (`data/injected/`, #19) covers this: every verifier also faces known injected errors.
2. It costs about 6× more per call than 3.5 Flash-Lite, mostly in thinking tokens.

**Options for the team:**

- **(a) Keep the default thinking level** (83%). This is what Tejas is using for the baseline.
- **(b) Pilot `thinking_level="low"`** (30 calls, free tier). If it lands in 60–80%, it's inside the target and roughly halves the study's cost.

## Verifier sanity check

The dedicated verifier (#32) was tried on two real gemini-3.6-flash answers to S04:

- On the **wrong** answer it **rejected** in 2 calls. It recomputed Fund B as $145,469.35 with the calculator and listed both wrong figures as issues.
- On a **correct** answer it **accepted** in 4 calls, after recomputing all three figures.

## Cost notes

- The free tier allows only **20 requests per day per model** for gemini-3.6-flash, so the pilot needed two days. The full study needs the paid tier, capped at **$20** in code (#29).
- Gemini sometimes reports no output tokens for a short reply. `llm.py` now logs a deliberately high estimate in that case, so the cap never undercounts (#36).
