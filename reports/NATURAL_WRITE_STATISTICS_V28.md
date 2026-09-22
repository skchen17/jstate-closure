# Natural Write Statistics — V28

On 25 calibration states, naturally generated outgoing fields were compared with their incoming fields. REC/Conv use direct tensor subtraction; KV uses the new slot minus the preceding slot, not a forced subtraction across unequal sequence lengths.

| channel | layer-state rows | median write norm | Q1 | Q3 |
|---|---:|---:|---:|---:|
| Conv | 150 | 233.666 | 224.453 | 246.816 |
| KV | 100 | 34.7573 | 31.8958 | 36.3623 |
| REC | 150 | 2.49255 | 2.15879 | 2.81145 |

These are descriptive native-field statistics, not state dimensionality or causal potency estimates. A small-write versus large-write matched causal control was not adjudicated.
