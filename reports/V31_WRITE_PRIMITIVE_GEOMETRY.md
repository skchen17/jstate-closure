# Write Primitive Geometry — V31

The exact architecture-valid REC+Conv contrast has 13,369,344 components. The frozen 100 development fit states × 10 TRAIN contrasts produce 1000 rows. Pooled descriptive r90/r95/r99 = 137/220/527; this is neither a model-state dimension nor a count of causally reusable primitives.

| TRAIN-token prefix | rows | r90 | r95 | r99 |
|---|---|---|---|---|
| 128 | 808 | 114 | 184 | 433 |
| 16 | 112 | 23 | 39 | 71 |
| 160 | 1000 | 137 | 220 | 527 |
| 32 | 224 | 39 | 65 | 133 |
| 64 | 424 | 67 | 109 | 239 |

Rank grows with token coverage. This alone cannot distinguish reusable composition from continually new causal directions. Full tensor matrix and bases stay off Git under `/data/CSK/J-space-project/v31-write-work`; spectrum and hashes are committed.
