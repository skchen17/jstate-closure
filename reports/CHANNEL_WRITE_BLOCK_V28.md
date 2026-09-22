# Channel Write Block — V28

| native condition | development median h1 Q | validation median h1 Q | dev gate | val gate |
|---|---:|---:|---:|---:|
| Conv | 37.146 | 40.811 | 1 | 1 |
| KV_last_slot_previous_copy | 2.356 | 2.354 | 1 | 1 |
| REC | 13.690 | 13.601 | 1 | 1 |
| REC+Conv | 34.552 | 36.818 | 1 | 1 |
| REC+Conv+KV_last_slot_previous_copy | 35.359 | 37.353 | 1 | 1 |

REC and Conv exact old-state blocks each matter. KV new-slot replacement is a different, length-preserving intervention. All are current-preserving only because the post-forward readout is fixed. Marginal effect magnitude is not a channel ranking.
