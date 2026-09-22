# Conv Layer Profile — V30

Exact native donor Conv fields were tested at every one of 24 recurrent layers on a prospectively frozen 10-state/role panel. All untouched cache fields are checked bitwise by the experiment. Full Conv: development cosine 0.921, L2 0.390; validation cosine 0.962, L2 0.280.

| layer | dev cos | dev L2 | val cos | val L2 |
|---|---|---|---|---|
| 0 | 0.270 | 0.963 | 0.171 | 0.988 |
| 1 | 0.156 | 0.989 | 0.055 | 1.004 |
| 2 | 0.091 | 0.997 | 0.084 | 0.997 |
| 4 | 0.073 | 0.999 | 0.052 | 0.999 |
| 5 | 0.139 | 0.990 | 0.151 | 0.989 |
| 6 | 0.062 | 0.999 | 0.055 | 0.998 |
| 8 | 0.044 | 1.000 | 0.013 | 1.001 |
| 9 | 0.028 | 1.000 | -0.017 | 1.002 |
| 10 | 0.046 | 0.999 | 0.030 | 1.000 |
| 12 | 0.069 | 0.998 | 0.116 | 0.995 |
| 13 | 0.114 | 0.995 | 0.186 | 0.989 |
| 14 | 0.108 | 0.994 | 0.062 | 0.999 |
| 16 | 0.083 | 0.997 | 0.100 | 0.996 |
| 17 | 0.293 | 0.975 | 0.303 | 0.978 |
| 18 | 0.328 | 0.961 | 0.276 | 0.970 |
| 20 | 0.306 | 0.983 | 0.351 | 0.974 |
| 21 | 0.195 | 0.989 | 0.218 | 0.990 |
| 22 | 0.264 | 0.986 | 0.238 | 0.988 |
| 24 | 0.121 | 0.994 | 0.124 | 0.995 |
| 25 | 0.144 | 0.993 | 0.192 | 0.989 |
| 26 | 0.070 | 0.998 | 0.030 | 1.001 |
| 28 | 0.252 | 0.975 | 0.274 | 0.975 |
| 29 | 0.142 | 0.992 | 0.192 | 0.985 |
| 30 | 0.431 | 0.932 | 0.488 | 0.906 |

Quartiles, halves, prefixes, suffixes, leave-quartile-out and 24 leave-single-out rows are preserved in `conv_layer_profile_*_v30.parquet`; a single layer's effect is not additive evidence of necessity.
