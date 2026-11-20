# Paired comparisons vs none

| condition_a | condition_b | n_tasks | n_pairs | n_dropped | n_draft_mismatch | success_a | success_b | delta | delta_ci_low | delta_ci_high | a_only_correct | b_only_correct | task_a_better | task_b_better | sign_test_p | note | model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | dedicated | 32 | 96 | 0 | 0 | 0.8854 | 0.9792 | 0.09375 | 0.03125 | 0.1771 | 0 | 9 | 0 | 6 | 0.03125 |  | gemini-3.6-flash |
| none | peer | 32 | 96 | 0 | 0 | 0.8854 | 0.8854 | 0 | -0.03125 | 0.03125 | 1 | 1 | 1 | 1 | 1 |  | gemini-3.6-flash |
| none | self | 32 | 96 | 0 | 0 | 0.8854 | 0.8854 | 0 | -0.03125 | 0.03125 | 1 | 1 | 1 | 1 | 1 |  | gemini-3.6-flash |
| none | weakened | 32 | 96 | 0 | 0 | 0.8854 | 0.875 | -0.01042 | -0.03151 | 0 | 1 | 0 | 1 | 0 | 1 |  | gemini-3.6-flash |

