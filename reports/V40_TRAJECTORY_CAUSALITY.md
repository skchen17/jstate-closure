# V40 — Trajectory Causality

Qwen early/middle/late REC groups are fixed thirds of its 24 recurrent layers.
Each factorial condition starts from recipient REC/KV with donor Conv as a common background; all future computation runs naturally.
The primary three-way contrast is on the externally labelled donor-versus-recipient answer margin.

| Role | Delay | Complete / sealed | Qualified | R000 donor recovery (all; qualified) | R111 donor recovery (all; qualified) | Three-way mean [95% CI] | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| development | 0 | 64/64 | 41 | 19/64; 11/41 | 21/64; 12/41 | 0.009 [-0.037, 0.050] | False |
| development | 8 | 64/64 | 42 | 6/64; 2/42 | 7/64; 2/42 | 0.007 [-0.030, 0.046] | False |
| validation | 0 | 32/32 | 22 | 11/32; 5/22 | 13/32; 7/22 | 0.010 [-0.052, 0.068] | False |
| validation | 8 | 32/32 | 20 | 5/32; 0/20 | 7/32; 2/20 | -0.056 [-0.113, 0.006] | False |
| independent_final | 0 | 32/32 | 19 | 5/32; 3/19 | 7/32; 5/19 | -0.018 [-0.056, 0.021] | False |
| independent_final | 8 | 32/32 | 22 | 3/32; 2/22 | 3/32; 2/22 | -0.017 [-0.057, 0.020] | False |

## Independent-final task-family interaction

| Delay | Family | Qualified states | Median three-way answer-margin interaction |
|---:|---|---:|---:|
| 0 | boolean_logic | 4 | 0.000 |
| 0 | long_context_dependency | 5 | 0.000 |
| 0 | state_transition | 2 | 0.016 |
| 0 | variable_binding | 8 | 0.000 |
| 8 | boolean_logic | 4 | -0.062 |
| 8 | long_context_dependency | 8 | 0.000 |
| 8 | state_transition | 3 | 0.031 |
| 8 | variable_binding | 7 | 0.000 |

## Final-answer hidden geometry at the last recurrent checkpoint

| Delay | Condition | Median cosine to native donor |
|---:|---|---:|
| 0 | R000 | 0.9515 |
| 0 | R100 | 0.9522 |
| 0 | R010 | 0.9546 |
| 0 | R001 | 0.9516 |
| 0 | R111 | 0.9546 |
| 8 | R000 | 0.9626 |
| 8 | R100 | 0.9625 |
| 8 | R010 | 0.9670 |
| 8 | R001 | 0.9623 |
| 8 | R111 | 0.9671 |

Hidden-state cosine summaries and checkpoint REC/Conv norms are in the sealed machine records.
Donor-like hidden geometry without task recovery is not task-grounded mediation.
This factorial tests depth composition on one channel background; it does not identify a unique primitive or all token-time mechanisms.
