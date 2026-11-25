# Seeded right-reason metrics

| condition | verifier_model | n_errors | n_controls | n_failed | n_missing_verdict | seeded_catch_rate | seeded_false_alarm_rate | right_reason | consequence_only | wrong_reason | abstained | right_reason_rate | right_reason_catch_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cross_family | openai/gpt-oss-20b | 17 | 14 | 0 | 0 | 0.9412 | 0.5714 | 16 | 0 | 0 | 0 | 1 | 0.9412 |
| dedicated | gemini-3.6-flash | 17 | 14 | 0 | 0 | 1 | 0.07143 | 16 | 0 | 1 | 0 | 0.9412 | 0.9412 |
| peer | gemini-3.6-flash | 17 | 14 | 0 | 0 | 1 | 0.07143 | 16 | 0 | 1 | 0 | 0.9412 | 0.9412 |
| self | gemini-3.6-flash | 17 | 14 | 0 | 1 | 1 | 0.2308 | 17 | 0 | 0 | 0 | 1 | 1 |
| weakened | gemini-3.6-flash | 17 | 14 | 0 | 0 | 0.9412 | 0.07143 | 16 | 0 | 0 | 0 | 1 | 0.9412 |

