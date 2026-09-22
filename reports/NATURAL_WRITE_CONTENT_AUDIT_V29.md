# Natural Write Content Audit — V29

REC and Conv write contrasts are exact `P_out,B − P_out,A` at all 24 recurrent layers; KV is only the newly appended token slot at all 8 attention layers. Pre-existing KV slots are bitwise identical between branches. Geometry is descriptive, not a carrier claim.

| role | channel | fields | median contrast norm | nonzero fraction |
|---|---|---:|---:|---:|
| development | Conv | 1800 | 63.901 | 1.000 |
| development | KV | 1200 | 26.504 | 1.000 |
| development | REC | 1800 | 0.892 | 1.000 |
| validation | Conv | 1200 | 64.195 | 1.000 |
| validation | KV | 800 | 26.129 | 1.000 |
| validation | REC | 1200 | 0.901 | 1.000 |

Train-only centered **joint REC+Conv contrast** spectrum has r90/r95/r99 = `11/22/55` over 90 training examples, rank `89` in ambient dimension `13369344`. Channel-specific full-vector centered spectra were not computed; per-field norms must not be substituted for them. The joint rank is not a dynamical state dimension.
