# Write Dose Response — V28

For fixed-shape REC/Conv, `P_in + α(P_out−P_in)` was written after natural token-t computation. h1 effect is measured against α=1 clean output.

| retained write α | development median Q | validation median Q |
|---:|---:|---:|
| 0 | 31.462240 | 36.671151 |
| 0.25 | 23.220543 | 26.822207 |
| 0.5 | 14.478242 | 16.721245 |
| 0.75 | 7.063916 | 7.642284 |
| 1 | 0.000000 | 0.000000 |

Both median curves decline monotonically. This is a finite dose relationship, not evidence of linearity, a compact state, or validity of KV interpolation.
