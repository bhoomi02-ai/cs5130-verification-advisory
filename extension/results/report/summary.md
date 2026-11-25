# Summary by condition

| condition | model | task_success | task_success_ci_low | task_success_ci_high | catch_rate | false_alarm_rate | mcc | fixed | broken | success_delta | mean_cost_usd | p95_latency_s | extra_cost_per_extra_correct | n_failed | n_missing_verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | gemini-3.6-flash | 0.8571 | 0.6429 | 1 |  |  |  | 0 | 0 | 0 | 0.01986 | 31.69 |  | 0 |  |
| dedicated | gemini-3.6-flash | 0.9762 | 0.9286 | 1 | 1 | 0 | 1 | 5 | 0 | 0.119 | 0.05178 | 76.16 | 0.2681 | 0 | 1 |
| self | gemini-3.6-flash | 0.8571 | 0.6429 | 1 | 0.3333 | 0 | 0.5477 | 0 | 0 | 0 | 0.03051 | 53.37 |  | 0 | 0 |
| weakened | gemini-3.6-flash | 0.8571 | 0.6429 | 1 | 0.5 | 0 | 0.6794 | 0 | 0 | 0 | 0.03361 | 89.97 |  | 0 | 0 |

