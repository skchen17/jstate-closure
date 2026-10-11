# V40 — Channel Function Profile

The primary readout is the externally generated task answer after one space token.
Each row uses all sealed task states and separately reports the native-pair-qualified subset.
Calibration and V39 outcomes are not counted below.

| Model | Role | Complete / sealed | Native qualified | Condition | Donor recovered / all sealed | Donor recovered / qualified | Recovery Δ vs BASE, bootstrap 95% CI |
|---|---|---:|---:|---|---:|---:|---:|
| Q | development | 64/64 | 41 | BASE | 5/64 | 0/41 | 0.000 [0.000, 0.000] |
| Q | development | 64/64 | 41 | REC_only | 5/64 | 0/41 | 0.000 [0.000, 0.000] |
| Q | development | 64/64 | 41 | Conv_only | 19/64 | 11/41 | 0.268 [0.146, 0.415] |
| Q | development | 64/64 | 41 | KV_only | 34/64 | 27/41 | 0.659 [0.512, 0.780] |
| Q | development | 64/64 | 41 | REC_Conv | 21/64 | 12/41 | 0.293 [0.171, 0.439] |
| Q | development | 64/64 | 41 | REC_Conv_KV | 49/64 | 41/41 | 1.000 [1.000, 1.000] |
| Q | development | 64/64 | 41 | DONOR_NATIVE | 49/64 | 41/41 | 1.000 [1.000, 1.000] |
| F | development | 64/64 | 18 | BASE | 9/64 | 0/18 | 0.000 [0.000, 0.000] |
| F | development | 64/64 | 18 | REC_only | 9/64 | 0/18 | 0.000 [0.000, 0.000] |
| F | development | 64/64 | 18 | Conv_only | 13/64 | 4/18 | 0.222 [0.056, 0.444] |
| F | development | 64/64 | 18 | KV_only | 15/64 | 5/18 | 0.278 [0.110, 0.500] |
| F | development | 64/64 | 18 | REC_Conv | 15/64 | 5/18 | 0.278 [0.111, 0.500] |
| F | development | 64/64 | 18 | REC_Conv_KV | 29/64 | 18/18 | 1.000 [1.000, 1.000] |
| F | development | 64/64 | 18 | DONOR_NATIVE | 29/64 | 18/18 | 1.000 [1.000, 1.000] |
| Q | validation | 32/32 | 22 | BASE | 5/32 | 0/22 | 0.000 [0.000, 0.000] |
| Q | validation | 32/32 | 22 | REC_only | 5/32 | 0/22 | 0.000 [0.000, 0.000] |
| Q | validation | 32/32 | 22 | Conv_only | 11/32 | 5/22 | 0.227 [0.090, 0.409] |
| Q | validation | 32/32 | 22 | KV_only | 18/32 | 13/22 | 0.591 [0.364, 0.818] |
| Q | validation | 32/32 | 22 | REC_Conv | 13/32 | 7/22 | 0.318 [0.136, 0.500] |
| Q | validation | 32/32 | 22 | REC_Conv_KV | 27/32 | 22/22 | 1.000 [1.000, 1.000] |
| Q | validation | 32/32 | 22 | DONOR_NATIVE | 27/32 | 22/22 | 1.000 [1.000, 1.000] |
| F | validation | 32/32 | 10 | BASE | 1/32 | 0/10 | 0.000 [0.000, 0.000] |
| F | validation | 32/32 | 10 | REC_only | 1/32 | 0/10 | 0.000 [0.000, 0.000] |
| F | validation | 32/32 | 10 | Conv_only | 5/32 | 3/10 | 0.300 [0.100, 0.600] |
| F | validation | 32/32 | 10 | KV_only | 7/32 | 5/10 | 0.500 [0.200, 0.800] |
| F | validation | 32/32 | 10 | REC_Conv | 7/32 | 5/10 | 0.500 [0.200, 0.800] |
| F | validation | 32/32 | 10 | REC_Conv_KV | 12/32 | 10/10 | 1.000 [1.000, 1.000] |
| F | validation | 32/32 | 10 | DONOR_NATIVE | 12/32 | 10/10 | 1.000 [1.000, 1.000] |
| Q | independent_final | 32/32 | 19 | BASE | 3/32 | 0/19 | 0.000 [0.000, 0.000] |
| Q | independent_final | 32/32 | 19 | REC_only | 3/32 | 0/19 | 0.000 [0.000, 0.000] |
| Q | independent_final | 32/32 | 19 | Conv_only | 5/32 | 3/19 | 0.158 [0.000, 0.316] |
| Q | independent_final | 32/32 | 19 | KV_only | 17/32 | 15/19 | 0.789 [0.579, 0.947] |
| Q | independent_final | 32/32 | 19 | REC_Conv | 7/32 | 5/19 | 0.263 [0.105, 0.474] |
| Q | independent_final | 32/32 | 19 | REC_Conv_KV | 21/32 | 19/19 | 1.000 [1.000, 1.000] |
| Q | independent_final | 32/32 | 19 | DONOR_NATIVE | 21/32 | 19/19 | 1.000 [1.000, 1.000] |
| F | independent_final | 32/32 | 8 | BASE | 7/32 | 0/8 | 0.000 [0.000, 0.000] |
| F | independent_final | 32/32 | 8 | REC_only | 7/32 | 0/8 | 0.000 [0.000, 0.000] |
| F | independent_final | 32/32 | 8 | Conv_only | 8/32 | 1/8 | 0.125 [0.000, 0.375] |
| F | independent_final | 32/32 | 8 | KV_only | 12/32 | 4/8 | 0.500 [0.125, 0.875] |
| F | independent_final | 32/32 | 8 | REC_Conv | 11/32 | 4/8 | 0.500 [0.125, 0.875] |
| F | independent_final | 32/32 | 8 | REC_Conv_KV | 16/32 | 8/8 | 1.000 [1.000, 1.000] |
| F | independent_final | 32/32 | 8 | DONOR_NATIVE | 16/32 | 8/8 | 1.000 [1.000, 1.000] |

REC+Conv+KV reconstructs the complete native donor cache and serves as a positive control; its recovery is not independent evidence of channel specialization.
REC+Conv versus Conv-only is a conditional contrast, not an independent REC-storage proof.
The source-fork and shuffled-state controls are summarized in the task-grounded report.


## Secondary answer-alphabet logit-vector fidelity

This exploratory operationalization computes `1 − distance(condition, native donor) / distance(BASE, native donor)` after mean-centering the frozen answer-alphabet logits. Zero is BASE and one is the native donor; it is **not** full-vocabulary or hidden-state fidelity and does not establish semantic storage.

| Model | Role | Identifiable / native qualified | REC mean [95% CI] | Conv mean [95% CI] | KV mean [95% CI] | REC+Conv mean [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| Q | development | 41/41 | 0.003 [-0.002, 0.008] | 0.289 [0.237, 0.339] | 0.536 [0.480, 0.596] | 0.298 [0.249, 0.351] |
| F | development | 18/18 | 0.002 [-0.008, 0.012] | 0.286 [0.190, 0.410] | 0.527 [0.402, 0.648] | 0.294 [0.192, 0.414] |
| Q | validation | 22/22 | 0.008 [0.001, 0.014] | 0.342 [0.270, 0.410] | 0.526 [0.433, 0.625] | 0.351 [0.279, 0.420] |
| F | validation | 10/10 | 0.015 [0.004, 0.028] | 0.413 [0.247, 0.599] | 0.644 [0.488, 0.774] | 0.437 [0.274, 0.611] |
| Q | independent_final | 19/19 | 0.011 [0.003, 0.019] | 0.300 [0.238, 0.361] | 0.622 [0.522, 0.719] | 0.314 [0.250, 0.378] |
| F | independent_final | 8/8 | 0.008 [-0.003, 0.019] | 0.365 [0.253, 0.532] | 0.535 [0.283, 0.755] | 0.404 [0.289, 0.553] |
