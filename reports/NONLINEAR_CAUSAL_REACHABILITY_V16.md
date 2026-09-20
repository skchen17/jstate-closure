# Empirical nonlinear causal reachability — V16

Frozen V13 captured h1 teacher J deltas are used **only as labels**, never as response-model/controller inputs. On V16 validation states, nearest measured residuals to the teacher J target are:

| design   |   states |   candidates |   action_count |   nearest_relative_residual |
|:---------|---------:|-------------:|---------------:|----------------------------:|
| dense    |       10 |        8.000 |          4.000 |                       5.076 |
| pair     |       10 |       16.000 |          2.000 |                       5.619 |
| single   |       50 |       58.000 |          1.000 |                       0.713 |
| triple   |       10 |        4.000 |          3.000 |                      10.005 |

Singles are one finite writeback. Pair, triple and dense entries are simultaneous same-state mixtures, **not** two-/four-token action sequences. This table alone cannot establish that multi-step nonlinear reachable sets exceed the V15 linear span. Every selected coordinate, action norm, candidate count and family is in `/data/CSK/J-space-project/jstate-closure/results/v16/processed/nonlinear_reachability_v16.parquet`.

In a separate **restricted actual teacher-forced sequence** audit, the same five validation states and same h4 teacher J target were used at all depths; each active step chooses one of ± two calibrated reliable primitives, without tangent transport. Exhaustive candidate counts and nearest residuals:

|   depth |   states |   candidate_count |   nearest_relative_residual |
|--------:|---------:|------------------:|----------------------------:|
|   1.000 |    5.000 |             4.000 |                       8.384 |
|   2.000 |    5.000 |            16.000 |                       3.549 |
|   4.000 |    5.000 |           256.000 |                       5.458 |

This is post-hoc nearest-set search, not an authorized response-model-based controller. The small four-primitive/five-state domain and unchanged raw directions prevent a general multi-step reachability claim. Per-state best sequence and action norm: `/data/CSK/J-space-project/jstate-closure/results/v16/processed/sequence_reachability_v16.parquet`.
