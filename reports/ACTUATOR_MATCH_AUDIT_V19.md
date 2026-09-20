# V19 realized probe actuator matching

Frozen thresholds: cosine ≥ 0.98; Pq/P0 norm ratio [0.9, 1.1]; each action readback cosine ≥ 0.95 and requested/realized gain within [0.8, 1.2].

h1 validation denominator 7360 response rows; matched 7360 (1.0000); mismatch 0. All mismatches remain in machine Parquet with `ACTION_REALIZATION_MISMATCH`. Train match rate 1.0000; validation match rate 1.0000.

## Readback distribution

| quantity | median matched | minimum matched | median mismatch |
|---|---|---|---|
| realized_action_pair_cosine | 1.0000 | 0.9997 | — |
| realized_norm_ratio_pq_over_p0 | 1.0000 | 0.9996 | — |
| realized_gain_p0 | 1.0000 | 0.9999 | — |
| realized_gain_pq | 1.0000 | 0.9996 | — |
| realized_delta_difference_norm | 1.8965 | 0.0000 | — |

## Median joint BF16 channel survival in both P0 and Pq

| channel | surviving requested elements |
|---|---|
| recurrent | 0.9976 |
| conv | 0.9988 |
| keys | 0.9445 |
| values | 0.9225 |

Requested q and a are separate. The action-match audit compares actual BF16 cache deltas in P0 and Pq across REC, Conv, K and V, not merely requested direction vectors.
