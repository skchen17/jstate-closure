# Finite-response linearity — V15

The frozen gate requires cosine ≥0.9, relative L2 ≤0.3, and norm ratio 0.7–1.3 for every SNR-qualified tested pair. Tests use one-sided effects relative to the clean state at ε=1: odd symmetry, additivity, homogeneity (factor 2), REC+Conv, and REC+Conv+KV. This is not the trivial algebraic oddness of a central difference definition.

| test        |   qualified |   pass_count |   median_cosine |   median_relative_l2 |   median_norm_ratio |
|:------------|------------:|-------------:|----------------:|---------------------:|--------------------:|
| additive    |          10 |            2 |           0.801 |                0.617 |               0.943 |
| homogeneous |          10 |            3 |           0.865 |                0.493 |               0.783 |
| odd         |          10 |            0 |           0.313 |                1.057 |               1.102 |
| rec_conv    |          10 |            1 |           0.816 |                0.578 |               0.778 |
| rec_conv_kv |          10 |            0 |           0.761 |                0.679 |               0.676 |

`FINITE_RESPONSE_LINEARITY_GATE = False`. The interface is repeatable in the limited retest, but not sufficiently linear under this operating rule. No V15 differential Hessian/secant geometry was promoted from V14. Machine rows: `finite_response_linearity_v15.parquet` and `finite_response_repeat_v15.parquet`.
