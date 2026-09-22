# Channel Write Content Factorial — V29

The seven frozen channel subsets form an exact native-field factorial: REC, Conv, KV, REC+Conv, REC+KV, Conv+KV, and complete REC+Conv+KV. The complete cache is an identity ceiling; the single Conv channel already passes. Therefore **no single channel sufficient / only combinations work** is false. Combinations improve fidelity but cannot be interpreted as additive main effects without interaction-aware analysis.

| channel | role | A←B cosine / magnitude / rel-L2 | B←A cosine / magnitude / rel-L2 | reciprocal gate |
|---|---|---:|---:|---:|
| REC | development | 0.062 / 0.095 / 0.999 | 0.259 / 0.132 / 0.980 | False |
| REC | validation | -0.002 / 0.095 / 1.003 | 0.358 / 0.137 / 0.962 | False |
| Conv | development | 0.988 / 0.971 / 0.163 | 0.992 / 0.989 / 0.127 | True |
| Conv | validation | 0.987 / 0.962 / 0.175 | 0.992 / 0.993 / 0.125 | True |
| KV | development | 0.082 / 0.067 / 0.997 | 0.108 / 0.067 / 0.995 | False |
| KV | validation | 0.088 / 0.067 / 0.996 | 0.093 / 0.070 / 0.996 | False |
| REC+Conv | development | 0.998 / 0.995 / 0.067 | 0.998 / 0.997 / 0.067 | True |
| REC+Conv | validation | 0.998 / 0.996 / 0.070 | 0.998 / 0.996 / 0.067 | True |
| REC+KV | development | 0.155 / 0.127 / 0.989 | 0.286 / 0.163 / 0.971 | False |
| REC+KV | validation | 0.121 / 0.125 / 0.993 | 0.342 / 0.175 / 0.962 | False |
| Conv+KV | development | 0.992 / 0.980 / 0.132 | 0.996 / 0.999 / 0.095 | True |
| Conv+KV | validation | 0.991 / 0.962 / 0.137 | 0.996 / 1.003 / 0.095 | True |
| REC+Conv+KV | development | 1.000 / 1.000 / 0.000 | 1.000 / 1.000 / 0.000 | True |
| REC+Conv+KV | validation | 1.000 / 1.000 / 0.000 | 1.000 / 1.000 / 0.000 | True |
