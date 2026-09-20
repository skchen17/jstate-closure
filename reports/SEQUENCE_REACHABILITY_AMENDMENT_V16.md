# V16 sequence-depth interpretation amendment

The original `sequence_reachability_v16.parquet` records **exactly** 1, 2 or 4 active actions. Those sets are not nested. This frozen, post-measurement correction takes their union for an **at-most**-depth comparison; it changes no action response, teacher label, split or historical record.

|   max_actions |   states |   candidate_count_at_most |   nearest_relative_residual_at_most |   median_best_exact_action_count |
|--------------:|---------:|--------------------------:|------------------------------------:|---------------------------------:|
|         1.000 |    5.000 |                     4.000 |                               8.384 |                            1.000 |
|         2.000 |    5.000 |                    20.000 |                               3.549 |                            2.000 |
|         4.000 |    5.000 |                   276.000 |                               3.549 |                            2.000 |

The five-state four-primitive search is still limited, and even the best at-most-four residual is far above the teacher target norm. It does not support general nonlinear controllability or authorize MPC. Per-state best exact count and sequence: `results/v16/processed/sequence_reachability_cumulative_v16.parquet` (SHA256 `72fe681d85100ea1a3280665bd148b6fa6836b0d2f6bc21aac340c9e8208cb42`). Correction freeze digest `0d5e681d29115718a25f52e00d0276e83292825ceee040bc568e89af23c10990`.
