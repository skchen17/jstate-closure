# Conv Depth Patterns — V31

TRAIN-only native write contrasts yield 24,000 Conv-layer energy records (1000 writes × 24 layers). Median relative energy by layer:

| Conv layer | median energy fraction | p90 fraction |
|---|---|---|
| 0 | 0.129 | 0.170 |
| 1 | 0.015 | 0.018 |
| 2 | 0.018 | 0.023 |
| 4 | 0.028 | 0.036 |
| 5 | 0.038 | 0.045 |
| 6 | 0.029 | 0.036 |
| 8 | 0.029 | 0.037 |
| 9 | 0.031 | 0.039 |
| 10 | 0.026 | 0.036 |
| 12 | 0.028 | 0.035 |
| 13 | 0.029 | 0.038 |
| 14 | 0.029 | 0.040 |
| 16 | 0.027 | 0.034 |
| 17 | 0.036 | 0.045 |
| 18 | 0.030 | 0.038 |
| 20 | 0.053 | 0.062 |
| 21 | 0.049 | 0.057 |
| 22 | 0.043 | 0.050 |
| 24 | 0.050 | 0.059 |
| 25 | 0.060 | 0.073 |
| 26 | 0.048 | 0.058 |
| 28 | 0.071 | 0.082 |
| 29 | 0.040 | 0.049 |
| 30 | 0.055 | 0.065 |

This norm profile is not a causal layer carrier. Separately, development and validation exact partial-transplant tables include single layers, quartiles, halves, prefixes, suffixes and leave-quartile-out controls. Neither development selected a passing small route nor did the fallback full24 meet the development gate.
