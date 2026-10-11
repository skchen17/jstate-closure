# V40 — Temporal Persistence

Qwen was primary. Each delay inserts fixed shared neutral tokens before the common task suffix.
Effects are empirical; no exponential decay law is assumed.

| Role | Delay | Complete / sealed | Native qualified | REC recovery (all; qualified) | Conv recovery (all; qualified) | KV recovery (all; qualified) | REC+Conv recovery (all; qualified) |
|---|---:|---:|---:|---:|---:|---:|---:|
| development | 0 | 64/64 | 41 | 5/64; 0/41 | 19/64; 11/41 | 34/64; 27/41 | 21/64; 12/41 |
| development | 1 | 64/64 | 44 | 4/64; 0/44 | 17/64; 13/44 | 35/64; 29/44 | 19/64; 15/44 |
| development | 2 | 64/64 | 44 | 4/64; 0/44 | 7/64; 3/44 | 41/64; 37/44 | 10/64; 6/44 |
| development | 4 | 64/64 | 44 | 4/64; 0/44 | 8/64; 2/44 | 43/64; 37/44 | 10/64; 4/44 |
| development | 8 | 64/64 | 42 | 4/64; 0/42 | 6/64; 2/42 | 43/64; 38/42 | 7/64; 2/42 |
| development | 16 | 64/64 | 38 | 4/64; 0/38 | 5/64; 2/38 | 35/64; 31/38 | 7/64; 3/38 |
| development | 32 | 64/64 | 27 | 5/64; 0/27 | 4/64; 1/27 | 29/64; 24/27 | 6/64; 2/27 |
| validation | 0 | 32/32 | 22 | 5/32; 0/22 | 11/32; 5/22 | 18/32; 13/22 | 13/32; 7/22 |
| validation | 1 | 32/32 | 22 | 5/32; 0/22 | 8/32; 3/22 | 16/32; 13/22 | 13/32; 8/22 |
| validation | 2 | 32/32 | 21 | 5/32; 0/21 | 7/32; 2/21 | 21/32; 17/21 | 8/32; 3/21 |
| validation | 4 | 32/32 | 21 | 5/32; 0/21 | 6/32; 1/21 | 23/32; 19/21 | 6/32; 1/21 |
| validation | 8 | 32/32 | 20 | 5/32; 0/20 | 5/32; 0/20 | 21/32; 17/20 | 7/32; 2/20 |
| validation | 16 | 32/32 | 19 | 6/32; 1/19 | 5/32; 1/19 | 19/32; 16/19 | 5/32; 1/19 |
| validation | 32 | 32/32 | 12 | 5/32; 0/12 | 4/32; 0/12 | 15/32; 11/12 | 4/32; 0/12 |
| independent_final | 0 | 32/32 | 19 | 3/32; 0/19 | 5/32; 3/19 | 17/32; 15/19 | 7/32; 5/19 |
| independent_final | 1 | 32/32 | 22 | 3/32; 0/22 | 9/32; 6/22 | 15/32; 13/22 | 9/32; 6/22 |
| independent_final | 2 | 32/32 | 22 | 1/32; 0/22 | 5/32; 4/22 | 20/32; 19/22 | 5/32; 4/22 |
| independent_final | 4 | 32/32 | 23 | 1/32; 0/23 | 2/32; 1/23 | 19/32; 18/23 | 3/32; 2/23 |
| independent_final | 8 | 32/32 | 22 | 1/32; 0/22 | 3/32; 2/22 | 20/32; 19/22 | 3/32; 2/22 |
| independent_final | 16 | 32/32 | 22 | 2/32; 0/22 | 2/32; 1/22 | 19/32; 18/22 | 2/32; 1/22 |
| independent_final | 32 | 32/32 | 14 | 3/32; 0/14 | 2/32; 0/14 | 14/32; 12/14 | 2/32; 0/14 |

Independent-final median answer-margin effect as a fraction of the contemporaneous native-donor ceiling:

| Delay | REC | Conv | KV | REC+Conv |
|---:|---:|---:|---:|---:|
| 0 | 0.011 | 0.382 | 0.711 | 0.386 |
| 1 | 0.040 | 0.319 | 0.607 | 0.352 |
| 2 | 0.022 | 0.188 | 0.783 | 0.224 |
| 4 | 0.028 | 0.167 | 0.828 | 0.186 |
| 8 | 0.032 | 0.123 | 0.787 | 0.177 |
| 16 | 0.022 | 0.124 | 0.760 | 0.161 |
| 32 | 0.000 | 0.062 | 0.914 | 0.074 |

Observed-grid half-crossing uses only states native-qualified at every delay and requires at least 16 such states.

| Role | Common qualified | Channel | First persistent half-crossing |
|---|---:|---|---:|
| development | 24 | Conv_only | 4 |
| development | 24 | KV_only | not identifiable |
| development | 24 | REC_Conv | 4 |
| development | 24 | REC_Conv_KV | not identifiable |
| development | 24 | REC_only | not identifiable |
| validation | 12 | Conv_only | not identifiable |
| validation | 12 | KV_only | not identifiable |
| validation | 12 | REC_Conv | not identifiable |
| validation | 12 | REC_Conv_KV | not identifiable |
| validation | 12 | REC_only | not identifiable |
| independent_final | 12 | Conv_only | not identifiable |
| independent_final | 12 | KV_only | not identifiable |
| independent_final | 12 | REC_Conv | not identifiable |
| independent_final | 12 | REC_Conv_KV | not identifiable |
| independent_final | 12 | REC_only | not identifiable |

A grid crossing is not a continuous or universal channel half-life.


## Independent-final task-family profiles (descriptive appendix)

Each cell uses the native-qualified states of that family at that delay; denominators may change with delay. These subgroup comparisons are exploratory.

| Family | Delay | Qualified / sealed | REC | Conv | KV | REC+Conv |
|---|---:|---:|---:|---:|---:|---:|
| boolean_logic | 0 | 4/8 | 0/4 | 0/4 | 4/4 | 0/4 |
| boolean_logic | 1 | 4/8 | 0/4 | 0/4 | 4/4 | 0/4 |
| boolean_logic | 2 | 4/8 | 0/4 | 2/4 | 4/4 | 0/4 |
| boolean_logic | 4 | 4/8 | 0/4 | 0/4 | 4/4 | 0/4 |
| boolean_logic | 8 | 4/8 | 0/4 | 1/4 | 4/4 | 1/4 |
| boolean_logic | 16 | 4/8 | 0/4 | 0/4 | 4/4 | 0/4 |
| boolean_logic | 32 | 4/8 | 0/4 | 0/4 | 4/4 | 0/4 |
| long_context_dependency | 0 | 5/8 | 0/5 | 3/5 | 2/5 | 5/5 |
| long_context_dependency | 1 | 8/8 | 0/8 | 6/8 | 0/8 | 6/8 |
| long_context_dependency | 2 | 8/8 | 0/8 | 2/8 | 5/8 | 4/8 |
| long_context_dependency | 4 | 8/8 | 0/8 | 1/8 | 6/8 | 2/8 |
| long_context_dependency | 8 | 8/8 | 0/8 | 1/8 | 7/8 | 1/8 |
| long_context_dependency | 16 | 8/8 | 0/8 | 1/8 | 6/8 | 1/8 |
| long_context_dependency | 32 | 2/8 | 0/2 | 0/2 | 2/2 | 0/2 |
| state_transition | 0 | 2/8 | 0/2 | 0/2 | 1/2 | 0/2 |
| state_transition | 1 | 2/8 | 0/2 | 0/2 | 1/2 | 0/2 |
| state_transition | 2 | 2/8 | 0/2 | 0/2 | 2/2 | 0/2 |
| state_transition | 4 | 4/8 | 0/4 | 0/4 | 2/4 | 0/4 |
| state_transition | 8 | 3/8 | 0/3 | 0/3 | 2/3 | 0/3 |
| state_transition | 16 | 3/8 | 0/3 | 0/3 | 2/3 | 0/3 |
| state_transition | 32 | 2/8 | 0/2 | 0/2 | 1/2 | 0/2 |
| variable_binding | 0 | 8/8 | 0/8 | 0/8 | 8/8 | 0/8 |
| variable_binding | 1 | 8/8 | 0/8 | 0/8 | 8/8 | 0/8 |
| variable_binding | 2 | 8/8 | 0/8 | 0/8 | 8/8 | 0/8 |
| variable_binding | 4 | 7/8 | 0/7 | 0/7 | 6/7 | 0/7 |
| variable_binding | 8 | 7/8 | 0/7 | 0/7 | 6/7 | 0/7 |
| variable_binding | 16 | 7/8 | 0/7 | 0/7 | 6/7 | 0/7 |
| variable_binding | 32 | 6/8 | 0/6 | 0/6 | 5/6 | 0/6 |

Bootstrap margin estimates for every family, delay, and channel are in the sealed machine appendix.
