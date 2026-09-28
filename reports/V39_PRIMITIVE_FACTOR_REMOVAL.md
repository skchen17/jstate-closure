# V39 — Primitive Factor Removal

Calibration-only, first-probe screen (20 states) used natural R111 and replaced each trace-eligible native factor by its seven-condition second-order prediction at one frozen layer; all descendants recomputed.

| Candidate | Layer | Median removed direction | Median remaining norm ratio | Families passing 0.50/0.70 |
|---|---:|---:|---:|---:|
| F1_POSTCONV_READ_FACTOR_C | 21 | 0.0940 | 0.9909 | 0/5 |
| F2_POSTCONV_UPDATE_FACTORS_X_B | 21 | 0.1898 | 0.9908 | 0/5 |
| F3_DECAY_CONTROL_DT_DA | 20 | 0.0838 | 1.0106 | 0/5 |
| F5_RECURRENT_UPDATE_DBX | 20 | 0.0746 | 1.0032 | 0/5 |

All four candidates fail the preregistered pilot gate; the frozen primary finalist is `null`. No formal development removal outcome was generated. This screen does not exclude multi-layer or other mechanisms.
