# Main-study results: analysis notes for team review

Status: draft for review (Dhanya). Source data: the `study-results` branch, main study `results/results.jsonl` (576 runs) and seeded reviews `results/seeded/` (155 reviews). Main results use the corrected S06 grading (`results/results_regraded.jsonl`, `results/report_regraded/`); the original grading (`results/report_original/`) is the sensitivity check. All numbers below come from those report tables unless stated otherwise. These notes are the basis for Section 7 (Results) and the slides; please comment on anything that looks wrong or over-claimed.

## 1. Data and integrity

- `scripts/check_study_data.py` on the final data: **OK**. 576 of 576 main runs and 155 of 155 seeded reviews; no duplicates, no malformed rows, no settings mismatches, and no shared-draft mismatches across all 96 task/repeat groups.
- One failed run: cross_family on T06-v003, repeat 2. Groq could not parse gpt-oss-20b's reply (it returned its reasoning as plain text) and the retry failed the same way. Failed runs are excluded from rates and counted separately, so cross_family has 95 scored runs.
- Re-grade: `scripts/regrade_results.py` changed exactly 2 of 576 rows (dedicated, T06-v003, repeats 2 and 3, final answer wrong to right) and no first-draft grades. Two team members ran it independently and got byte-identical files.

## 2. Task success and paired comparisons (main result)

First drafts were 88.5% correct (85 of 96), so there were 11 wrong drafts to catch, on six tasks (S04, S06, T04-v003, T06-v003, T06-v004, T08-v004).

| condition | task success | 95% CI | change vs none (points) | 95% CI | tasks better / worse | sign test p | Holm p |
|---|---|---|---|---|---|---|---|
| none | 88.5% (85 of 96) | [79.2, 95.8] | | | | | |
| self | 88.5% (85 of 96) | [78.1, 96.9] | 0.0 | [−3.1, +3.1] | 1 / 1 | 1.00 | 1.00 |
| peer | 88.5% (85 of 96) | [78.1, 96.9] | 0.0 | [−3.1, +3.1] | 1 / 1 | 1.00 | 1.00 |
| dedicated | 100% (96 of 96) | degenerate at 100 | **+11.5** | **[+4.2, +20.8]** | 6 / 0 | 0.031 | 0.156 |
| weakened | 87.5% (84 of 96) | [77.1, 95.8] | −1.0 | [−3.2, 0.0] | 0 / 1 | 1.00 | 1.00 |
| cross_family | 90.5% (86 of 95) | [81.3, 96.9] | +1.0 | [−4.2, +6.3] | 2 / 2 | 1.00 | 1.00 |

- Only dedicated improved task success. Its interval excludes zero, but the Holm-adjusted sign test over the five planned comparisons (m = 5) is not significant (p = 0.156). With 6 tasks better and 0 worse, 0.031 is the smallest p the sign test can give.
- dedicated's bootstrap interval collapses at 100% because every run is correct; report it as "100% (96 of 96); the interval is degenerate at this ceiling", not as certainty.
- cross_family's success is 90.5% (86 of 95) as a share of scored runs, but 89.6% in the paired comparison, which is task-weighted over matched pairs (Section 6.7). The two differ only because of the one failed run; both are correct.

## 3. What happened to the wrong drafts

| condition | caught | missed | caught and fixed | caught but not fixed | catch-fix rate | broken | human review |
|---|---|---|---|---|---|---|---|
| self | 6 of 11 | 5 | 1 | 5 | 17% (1 of 6) | 1 | 5.2% |
| peer | 6 of 11 | 5 | 1 | 5 | 17% (1 of 6) | 1 | 7.3% |
| dedicated | 11 of 11 | 0 | 11 | 0 | 100% (11 of 11) | 0 | 1.0% |
| weakened | 5 of 11 | 6 | 0 | 5 | 0% (0 of 5) | 1 | 6.3% |
| cross_family | 6 of 9 | 3 | 3 | 3 | 50% (3 of 6) | 2 | 11.6% |

("caught" uses the first verdict, reject or abstain; cross_family has 9 wrong drafts with a readable verdict after the failed run and one missing verdict.)

- **Finding an error is not fixing it.** self and peer caught 6 of 11 but fixed only 1; weakened caught 5 and fixed none.
- **The calculator is what makes the difference.** dedicated and weakened use the same model and the same review prompt and differ only in the calculator: 11 of 11 caught and all fixed, versus 5 of 11 caught and none fixed.
- dedicated broke no correct answers; every other verifier broke at least one, cross_family two.

## 4. Verifier metrics (first verdict, main study)

| condition | catch rate | false-alarm rate | MCC |
|---|---|---|---|
| self | 54.5% (6 of 11) | 2.4% (2 of 85) | 0.60 |
| peer | 54.5% (6 of 11) | 1.2% (1 of 85) | 0.65 |
| dedicated | 100% (11 of 11) | 2.4% (2 of 84) | 0.91 |
| weakened | 45.5% (5 of 11) | 1.2% (1 of 85) | 0.58 |
| cross_family | 66.7% (6 of 9) | **16.5% (14 of 85)** | 0.36 |

- cross_family flags far more correct drafts than any Gemini verifier (16.5% versus 1–2%), which also explains its high human-review rate and its two broken answers.

## 5. Seeded set and caught with the right reason

| condition | seeded catch rate | false alarms on controls | right reason (rule) | right reason (hand-checked) |
|---|---|---|---|---|
| self | 100% (17 of 17) | 23.1% (3 of 13) | 17 of 17 | 17 of 17 |
| peer | 100% (17 of 17) | 7.1% (1 of 14) | 16 of 17 | 17 of 17 |
| dedicated | 100% (17 of 17) | 7.1% (1 of 14) | 16 of 17 | 17 of 17 |
| weakened | 94.1% (16 of 17) | 7.1% (1 of 14) | 16 of 16 | 16 of 16 |
| cross_family | 94.1% (16 of 17) | **57.1% (8 of 14)** | 16 of 16 | **12 of 16** |

- The seeded errors are near the ceiling: every verifier caught 94–100%, so the seeded set confirms the verifiers can find planted errors but barely separates them. The controls separate them more: cross_family raised false alarms on 8 of 14 correct answers.
- **Hand check of the right-reason rule:** we checked all 83 right-reason decisions on rejected seeded errors and agreed with the rule on 77 (93%). The rule erred in both directions: it missed two correct explanations of an infeasible status that used only one matching term (dedicated and peer on S10), and it credited four cross_family reviews that quoted the wrong figure but misdiagnosed the error (S04 entry fee, S06 payment, S06 withdrawal, S08 cash). Sheet: `results/audit/right_reason_hand_audit.csv` on `study-results`.
- After the hand check, every Gemini verifier named the planted error in every catch; cross_family did so in 12 of 16 (75%).

## 6. Cost and latency

| condition | mean cost per run | p95 latency | extra cost per extra correct answer vs none |
|---|---|---|---|
| none | $0.02 | 38 s | |
| self | $0.03 | 84 s | (no gain) |
| peer | $0.03 | 121 s | (no gain) |
| dedicated | $0.05 | 70 s | $0.26 |
| weakened | $0.03 | 104 s | (no gain) |
| cross_family | $0.02 | 195 s | $0.27 |

- dedicated costs about 2.6 times none per run but is the only condition that buys extra correct answers, at about $0.26 each. cross_family's cost uses Groq's list price (we ran on the free tier).
- Total spend: [TBD from scripts/spend.py, excluding the follow-up experiment].

## 7. Sensitivity check: original S06 grading

Every condition except dedicated is identical under the original grading. dedicated: 97.9% (94 of 96) [93.8, 100], +9.4 points [+3.1, +17.7] vs none, 6 tasks better and 0 worse, sign test p = 0.031, Holm p = 0.156, 9 fixed, $0.32 per extra correct answer. The S06 correction changes the size of dedicated's improvement, not the conclusion.

## 8. Main deductions

1. Only a verifier with a calculator improved task success (+11.5 points [+4.2, +20.8]); the effect is consistent (6 tasks better, 0 worse) but not significant after Holm correction across five comparisons.
2. Without the calculator, verifiers find about half the arithmetic errors and almost never fix them: catching is not fixing.
3. Self-checking and peer review with the same model gave no gain over no verification.
4. The cross-family verifier catches some errors but raises many false alarms (16.5% of correct drafts, 57% of seeded controls), breaks correct answers, sends the most runs to human review, and misdiagnoses a quarter of the seeded errors it catches.
5. The seeded set is too easy to rank the verifiers on catching; its controls and the right-reason check are more informative.

## 9. Caveats to carry into the report

- 32 tasks × 3 repeats and only 11 organic wrong drafts on six tasks: intervals are wide and most comparisons cannot detect small effects.
- The S06 grading correction was made after seeing results; it is applied identically to all six settings and reported with the original grading as a sensitivity check.
- Right-reason matching is a heuristic; the hand check above is its validation.
- The follow-up experiment on the 14 unused variants (#67) is separate and exploratory; none of its runs are in these numbers.
