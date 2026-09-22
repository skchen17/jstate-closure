# Write Rank Scaling — V30

Centered pooled natural REC+Conv contrast ranks are descriptive, not causal-state dimensions. The 90-state×8-pair original fit had r90/r95/r99 = 74/131/320.

Balanced state scaling (8 TRAIN pairs/state):

| states | rows | r90 | r95 | r99 |
|---|---|---|---|---|
| 10 | 80 | 19 | 31 | 58 |
| 120 | 960 | 75 | 135 | 354 |
| 20 | 160 | 36 | 58 | 115 |
| 40 | 320 | 65 | 104 | 219 |
| 80 | 640 | 74 | 128 | 305 |

Original-fit token coverage scaling (90 states):

| pairs/state | rows | r90 | r95 | r99 |
|---|---|---|---|---|
| 1 | 90 | 12 | 24 | 54 |
| 2 | 180 | 23 | 45 | 100 |
| 4 | 360 | 41 | 78 | 179 |
| 8 | 720 | 74 | 131 | 320 |

The 120-state point uses 30 extra frozen development states and deterministic prewrite-eligible TRAIN pairs selected **after development response observation**; it is explicitly supplemental and was never used for any causal basis, transport fit, gate or finalist. State r95 growth slows after 80, but token coverage 1→8 still grows 24→131; the exact write geometry is not shown saturated. Behavioral k64 failure is a separate causal observation.
