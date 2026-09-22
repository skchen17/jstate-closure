# Local Write Geometry — V30

For each target, a centered local REC+Conv basis was fit only from its 70–72 eligible TOKEN_TRAIN contrasts. k=8/16/32/64 was tested; held-out token writes were not used to fit it. The 32D source–target principal-angle audit is descriptive, not a causal language proof.

| role | states | median r90 | median r95 | median r99 | median 32D principal cosine | median max angle ° | median chordal |
|---|---|---|---|---|---|---|---|
| development | 30 | 45.000 | 54.000 | 65.000 | 0.986 | 41.463 | 1.377 |
| validation | 60 | 45.000 | 54.000 | 65.000 | 0.986 | 48.548 | 1.412 |

The local TRAIN spectrum can have r95 ≈54 while the unseen-token k64 natural-write reconstruction median relative L2 remains 0.557 development and 0.558 validation. Thus TRAIN geometry does not establish unseen-token sufficiency.
