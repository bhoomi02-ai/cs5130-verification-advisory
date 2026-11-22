# Final report draft: Methods and Limitations

> Draft for the team to edit (Methods and Limitations by Bhoomika, Architecture by Tejas). Numbers marked **[pending]** wait for the last `cross_family` runs and its seeded reviews. Numbers use the **corrected S06 grading** (PR #63) as the main result and the original grading as the sensitivity check; if the team decides otherwise, swap them.

## Architecture and pipeline

> The study design (conditions, grading, statistics) is in Methods. This section covers how the runs were executed and recorded. Figure 1 shows the whole pipeline.

### Components
- **Scenarios** (`data/scenarios/`): one JSON file per task with the portfolio, policy, request and answer key; `data/study_tasks.txt` lists the 32 study tasks.
- **Agents** (`conditions.py`): the advisor and each condition's verifier, built from versioned prompts.
- **Calculator:** a sandboxed tool the verifiers with a calculator can call to recompute numbers.
- **Model gateway** (`llm.py`): every model call, Gemini or Groq, goes through one wrapper that logs tokens, cost and latency and stops at the $20 cap.
- **Checker** (`checker.py`): grades answers against the answer key, which agents never see.

### One run
- `scripts/run_condition.py` runs one condition over the task list with fixed model settings.
- For each task and run, the advisor's first draft is written once, saved, and reused by every condition.
- The condition's verifier reviews the draft; if it rejects, the advisor revises once.
- The checker grades both the first draft and the final answer.
- One row is added to `results/results.jsonl` with a run ID, the verdict, both grades, tokens, cost and time.

### Failures and reruns
- A call that still fails after retries is saved as a failed run, so nothing is silently lost.
- A rerun skips finished runs, so an interrupted batch can simply be restarted.
- `--retry-failed` reruns failed runs and replaces their rows instead of adding duplicates.
- Three failures in a row stop the script; a quota or code error stops it without saving.
- The script refuses to mix runs with different model settings in one results file.
- Usage is counted per attempt, so retries show up in the cost.

### Keeping the data trustworthy
- Results are kept on the `study-results` branch, off `main`.
- A read-only completeness check (`scripts/check_study_data.py`) confirms every task × condition × run appears exactly once, with the same first draft and settings across conditions.
- The S06 regrade (`scripts/regrade_results.py`) writes a separate file and never edits the original results.

### How the code was built
- Each change went through an issue, its own branch and a pull request.
- A PR was merged only after a teammate reviewed it and the full test suite (about 480 tests) passed.
- All study runs used the merged code on `main`.

## Methods

### Task set

The study uses 32 tasks (`data/study_tasks.txt`): the ten hand-built scenarios S01–S10 (rebalancing under policy constraints, cash needs, policy audits, fund comparisons after fees, retirement projections, and two infeasible requests) and 22 generated variants of them (two per S01–S08 family, three each for S09 and S10). Each scenario gives the advisor a portfolio, a client policy and a request, plus an output contract (`NEUTRAL_OUTPUT_V1`). Agents never see the answer key or the checks.

### Advisor and conditions

One advisor model, `gemini-3.6-flash`, wrote every answer with the shared prompt `advisor-v1` (temperature 0.7, top-p 0.9, default thinking). Each task ran three times. The advisor's **first draft** for a (task, run) pair was written once and reused by every condition, so differences between conditions come from verification alone.

Six conditions were compared:

| Condition | Verifier |
|---|---|
| None | no review; the draft is the final answer |
| Self | the advisor reviews its own answer |
| Peer | an independent call of the same model reviews it |
| Dedicated | a separate verifier role with a sandboxed calculator (up to 3 calculator rounds before its verdict) |
| No calculator (ablation) | the dedicated verifier without the calculator |
| Cross-family (ablation) | the dedicated verifier played by `openai/gpt-oss-20b` on Groq (low reasoning effort), with the calculator |

A verifier returns accept, reject or abstain. Accept keeps the draft. Abstain, or a verdict that cannot be read, sends the run to **human review** with the draft as the final answer. Reject makes the advisor revise once; the same verifier then checks the revision: accept makes it the final answer (`revised`), anything else sends it to human review (with the revision as the final answer).

### Grading

A deterministic checker (`checker.py`) grades the first draft and the final answer of every run from the scenario's numbers and the answer key: schema, required metrics, status, independent recomputation of the results, and a replay of any trades against the policy. Grading never uses a model.

**S06 scoring fix.** After the main runs we found that the checker asked more of the S06-family tasks (S06, T06-v003, T06-v004 in the study) than the tasks do: the client asks only for the extra monthly payment "rounded upward to cents", but the checker also required the unrounded payment. The team corrected the rule for the whole family (PR #63), applied it identically to all six conditions, and re-graded the saved answers (`scripts/regrade_results.py`). Exactly two grades changed, both Dedicated final answers on T06-v003 (runs 2 and 3, wrong → right); no first draft changed, so the baseline and the pairing are unaffected. Because the fix was found after seeing the results and it helps one condition, we report the **original grading as a sensitivity check** next to every headline number.

### Seeded error set

To test the verifiers on known errors, 31 hand-made answers (17 with one known error of six types — arithmetic, grounding, policy violation, unit, violation list, wrong status — and 14 correct controls) were each reviewed once by every verifier, with no revision. A review counts as **caught for the right reason** when its issues name the actual error (`right_reason.py`). These are sanity checks of the verifiers, not results about the advisory system.

### Metrics

- **Task success:** share of runs whose final answer the checker grades right.
- **Catch rate / false-alarm rate:** of the drafts that were wrong (right), the share the verifier rejected; **MCC** summarises both.
- **Fixed / broken:** wrong draft → right final answer, and right draft → wrong final answer.
- **Human review:** reported separately by route (abstain, unreadable verdict, revision not accepted); task success is reported both with and without these runs.
- **Cost:** tokens, dollars and latency per run (each run includes its share of the shared draft), and extra cost per extra correct answer relative to None.

Runs whose API call failed after retries are left out of every rate and reported as failed.

### Statistics

The primary result is the **paired change in task success against None** with a 95% bootstrap confidence interval over tasks (1,000 resamples; repeats of a task are averaged, not treated as independent). As a secondary test we report an exact two-sided sign test over tasks (ties dropped), both raw and **Holm-adjusted over the five planned comparisons with None** (Self, Peer, Dedicated, No calculator, Cross-family; the family is fixed at five even while a condition is unfinished, so the adjusted p does not depend on which runs are done). The proposal predicted that the dedicated, tool-equipped verifier would help most, but it did not name a single primary test, so we quote the adjusted p-value. Confidence intervals are not adjusted for multiple comparisons.

### Cost control

Every model call went through one wrapper (`llm.py`) that logs tokens, cost and latency and enforces a $20 spending cap. Total spend was about **$12.1** of the $20 cap ([pending] final figure). The cross-family verifier ran on Groq's free tier.

## Limitations

1. **One advisor model and one prompt.** All findings are for `gemini-3.6-flash` with `advisor-v1`; another advisor, with different errors, could rank the verifiers differently.
2. **Smaller than planned.** The proposal planned about 40 tasks × 5 runs; budget and time gave 32 tasks × 3 runs (96 runs per condition). With an 88.5% baseline, only **11 first drafts were wrong**, so each catch rate rests on 11 drafts (one catch moves it by about 9 points), and the confidence intervals are wide.
3. **Errors concentrate in a few tasks.** The 11 wrong first drafts came from six tasks: S04, S06, T06-v003 and T06-v004 (compounding-arithmetic slips) and one each on T04-v003 and T08-v004. The study says little about verifiers on other kinds of mistakes.
4. **Significance is fragile.** Dedicated improved on None on 6 tasks and was worse on none; 0.031 is the smallest p a sign test can give with 6 tasks, and Holm-adjusted over the five comparisons with None it is 0.156. The bootstrap interval excludes zero but is unadjusted. We report the effect size and interval as the main result.
5. **A scoring rule was corrected after the results were seen.** The S06 fix is justified by the task wording, applied to all conditions and changes only two runs, but it was found post hoc; the original grading is reported as a sensitivity check (Dedicated 97.9% vs 100%).
6. **The seeded errors were easier than the real ones.** Every verifier caught 94–100% of the seeded errors, but on the advisor's own errors Self and Peer caught 6 of 11, No calculator 5 of 11 and Dedicated 11 of 11. Single, visible slips overstate what a verifier without a calculator can do; the seeded set is a sanity check only.
7. **Cross-family ran on Groq's free tier.** Rate limits stretched the runs over several days, and **4 of its 96 runs failed**, all on S06-family tasks: gpt-oss-20b's replies could not be parsed by Groq on three of them, and one hit a network error ([pending] after the final retry). Its numbers rest on fewer runs, and the failures cluster on the hardest arithmetic tasks.
8. **Advisor and verifier share a model.** In Self, Peer, Dedicated and No calculator the advisor and the verifier are the same model, so their errors may be correlated; only Cross-family uses a different model family, and only for the verifier (the revision is still Gemini).
9. **Synthetic tasks and a strict checker.** The scenarios are synthetic and the checker demands exact numbers within small tolerances; a human advisor might accept answers it rejects, and vice versa.
10. **Human review is not simulated.** Runs sent to human review keep the draft or revision as their final answer; a real reviewer could fix (or miss) them. We therefore report task success with and without these runs.
11. **One revision round.** A rejected answer is revised once and checked once more; a longer loop might fix more answers (or break more), so our results are for a single round.
12. **Cost may not compare across providers.** Cost is computed from each provider's list price per token (Gemini, and Groq's $0.075 / $0.30 per million tokens for gpt-oss-20b), not from what we were billed: the Cross-family verifier ran on Groq's free tier and cost us nothing. Prices also differ between providers for reasons unrelated to the verifier, so Cross-family's cost is less comparable than the others'.
13. **Runs were spread over several days.** The study runs ran from Nov 19 to Nov [pending]; hosted models can be updated behind a fixed model name, so conditions run on different days may have met slightly different models. The shared first drafts limit this to the verifier and revision calls.
