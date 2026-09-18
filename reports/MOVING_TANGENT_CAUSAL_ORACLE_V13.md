# Static and moving-tangent causal oracle — V13

| method                             |   horizon |   direction |   magnitude |   semantic_continuous |   semantic_legacy |   output |   sign | gate_pass   |
|:-----------------------------------|----------:|------------:|------------:|----------------------:|------------------:|---------:|-------:|:------------|
| global_causal_basis_oracle         |         1 |       0.097 |       6.111 |                 0.097 |             0.032 |    0.133 |  0.520 | False       |
| global_causal_basis_oracle         |         2 |       0.102 |       5.522 |                 0.102 |             0.032 |    0.154 |  0.600 | False       |
| global_causal_basis_oracle         |         4 |       0.162 |       3.909 |                 0.162 |             0.012 |    0.231 |  0.625 | False       |
| global_causal_basis_oracle         |         8 |       0.279 |       2.128 |                 0.279 |             0.012 |    0.373 |  0.680 | False       |
| global_pca_oracle                  |         1 |       0.256 |       0.461 |                 0.256 |             0.076 |    0.296 |  0.760 | False       |
| global_pca_oracle                  |         2 |       0.356 |       0.644 |                 0.356 |             0.056 |    0.352 |  0.480 | False       |
| global_pca_oracle                  |         4 |       0.418 |       0.858 |                 0.418 |             0.080 |    0.401 |  0.583 | False       |
| global_pca_oracle                  |         8 |       0.494 |       0.957 |                 0.494 |             0.084 |    0.526 |  0.560 | False       |
| moving_tangent_interpolated_oracle |         1 |       0.772 |       0.871 |                 0.772 |             0.356 |    0.797 |  0.760 | False       |
| moving_tangent_interpolated_oracle |         2 |       0.689 |       0.897 |                 0.689 |             0.192 |    0.669 |  0.680 | False       |
| moving_tangent_interpolated_oracle |         4 |       0.590 |       0.994 |                 0.590 |             0.148 |    0.567 |  0.792 | False       |
| moving_tangent_interpolated_oracle |         8 |       0.513 |       0.965 |                 0.513 |             0.124 |    0.546 |  0.640 | False       |
| raw_teacher_intervention           |         1 |       1.000 |       1.000 |                 1.000 |             1.000 |    1.000 |  1.000 | True        |
| raw_teacher_intervention           |         2 |       1.000 |       1.000 |                 1.000 |             1.000 |    1.000 |  1.000 | True        |
| raw_teacher_intervention           |         4 |       1.000 |       1.000 |                 1.000 |             1.000 |    1.000 |  1.000 | True        |
| raw_teacher_intervention           |         8 |       1.000 |       1.000 |                 1.000 |             1.000 |    1.000 |  1.000 | True        |
| static_local_causal_oracle         |         1 |       0.739 |       0.825 |                 0.739 |             0.324 |    0.757 |  0.560 | False       |
| static_local_causal_oracle         |         2 |       0.645 |       0.874 |                 0.645 |             0.128 |    0.656 |  0.800 | False       |
| static_local_causal_oracle         |         4 |       0.578 |       0.966 |                 0.578 |             0.104 |    0.551 |  0.708 | False       |
| static_local_causal_oracle         |         8 |       0.500 |       0.984 |                 0.500 |             0.124 |    0.534 |  0.600 | False       |

Frozen moving method / alpha: `moving_tangent_interpolated_oracle` / `1.0`. Development panel: `10`; independent final panel: `25`. Moving tangent improves static: `True`. All h1/h2/h4/h8 gates pass: `False`.
