# Outcomes by condition

| condition | model | task_success | task_success_excl_human_review | human_review_rate | task_success_excl_human_review_ci_low | task_success_excl_human_review_ci_high | kept_correct | fixed | broken | still_wrong | missed | caught_fixed | caught_not_fixed | caught_escalated | catch_fix_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | gemini-3.6-flash | 0.8854 | 0.8854 | 0 | 0.7917 | 0.9583 | 85 | 0 | 0 | 11 |  |  |  |  |  |
| cross_family | gemini-3.6-flash | 0.9053 | 0.9405 | 0.1158 | 0.8889 | 0.9889 | 83 | 3 | 2 | 7 | 3 | 3 | 3 | 0 | 0.5 |
| self | gemini-3.6-flash | 0.8854 | 0.9341 | 0.05208 | 0.8387 | 0.9892 | 84 | 1 | 1 | 10 | 5 | 1 | 5 | 0 | 0.1667 |
| peer | gemini-3.6-flash | 0.8854 | 0.9438 | 0.07292 | 0.8722 | 0.9889 | 84 | 1 | 1 | 10 | 5 | 1 | 5 | 0 | 0.1667 |
| dedicated | gemini-3.6-flash | 1 | 1 | 0.01042 | 1 | 1 | 85 | 11 | 0 | 0 | 0 | 11 | 0 | 0 | 1 |
| weakened | gemini-3.6-flash | 0.875 | 0.9333 | 0.0625 | 0.8226 | 0.9785 | 84 | 0 | 1 | 11 | 6 | 0 | 5 | 0 | 0 |

task_success_excl_human_review excludes runs the verifier escalated; read it together with human_review_rate.
