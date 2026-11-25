# Outcomes by condition

| condition | model | task_success | task_success_excl_human_review | human_review_rate | task_success_excl_human_review_ci_low | task_success_excl_human_review_ci_high | kept_correct | fixed | broken | still_wrong | missed | caught_fixed | caught_not_fixed | caught_escalated | catch_fix_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | gemini-3.6-flash | 0.8571 | 0.8571 | 0 | 0.6429 | 1 | 36 | 0 | 0 | 6 |  |  |  |  |  |
| dedicated | gemini-3.6-flash | 0.9762 | 1 | 0.07143 | 1 | 1 | 36 | 5 | 0 | 1 | 0 | 5 | 1 | 0 | 0.8333 |
| self | gemini-3.6-flash | 0.8571 | 0.9 | 0.04762 | 0.6429 | 1 | 36 | 0 | 0 | 6 | 4 | 0 | 2 | 0 | 0 |
| weakened | gemini-3.6-flash | 0.8571 | 0.9231 | 0.07143 | 0.7692 | 1 | 36 | 0 | 0 | 6 | 3 | 0 | 3 | 0 | 0 |

task_success_excl_human_review excludes runs the verifier escalated; read it together with human_review_rate.
