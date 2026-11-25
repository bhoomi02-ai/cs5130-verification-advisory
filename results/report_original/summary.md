# Summary by condition

| condition | model | task_success | task_success_ci_low | task_success_ci_high | catch_rate | false_alarm_rate | mcc | fixed | broken | success_delta | mean_cost_usd | p95_latency_s | extra_cost_per_extra_correct | n_failed | n_missing_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | gemini-3.6-flash | 0.8854 | 0.7917 | 0.9583 |  |  |  | 0 | 0 | 0 | 0.01894 | 38.43 |  | 0 |  |
| cross_family | gemini-3.6-flash | 0.9053 | 0.8125 | 0.9688 | 0.6667 | 0.1647 | 0.3609 | 3 | 2 | 0.01053 | 0.02436 | 194.6 | 0.2732 | 1 | 1 |
| self | gemini-3.6-flash | 0.8854 | 0.7812 | 0.9688 | 0.5455 | 0.02353 | 0.6015 | 1 | 1 | 0 | 0.03178 | 84.11 |  | 0 | 0 |
| peer | gemini-3.6-flash | 0.8854 | 0.7812 | 0.9688 | 0.5455 | 0.01176 | 0.6538 | 1 | 1 | 0 | 0.03374 | 120.6 |  | 0 | 0 |
| dedicated | gemini-3.6-flash | 0.9792 | 0.9375 | 1 | 1 | 0.02381 | 0.9088 | 9 | 0 | 0.09375 | 0.04893 | 69.86 | 0.3199 | 0 | 1 |
| weakened | gemini-3.6-flash | 0.875 | 0.7708 | 0.9583 | 0.4545 | 0.01176 | 0.5826 | 0 | 1 | -0.01042 | 0.03222 | 104.1 |  | 0 | 0 |

