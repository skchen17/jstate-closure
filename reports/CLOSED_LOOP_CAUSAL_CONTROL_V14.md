# Closed-loop finite causal control — V14 development

The tested controller uses a shared rank-9 REC/conv/KV direction basis, finite central response at ε=1 (because exact-JVP equivalence failed), ridge `0.01`, trust radius `0.5`, four steps, and backtracking `[1,.5,.25]`. At every accepted step the causal response is remeasured and the nearest V13 tangent atlas basis is retrieved. Only `1` atlas index was actually visited, so this run **did not test an effective between-basis transport switch**. It optimizes weighted future J, logits, continuous semantics, and workspace effects, **not raw target-state distance**. Horizon objectives were h1 and h1+h2+h4+h8.

Static V13, V13 moving-interpolated, and V14 closed-loop on the same validation panel:

| method                                  |   horizon |   direction |   magnitude |   output |   semantic_legacy |   semantic_continuous |   sign |
|:----------------------------------------|----------:|------------:|------------:|---------:|------------------:|----------------------:|-------:|
| closed_loop_finite_response_h1          |         1 |       0.699 |       0.981 |    0.671 |             0.430 |                 0.699 |  0.800 |
| closed_loop_finite_response_h1          |         2 |       0.535 |       1.005 |    0.514 |             0.110 |                 0.535 |  0.800 |
| closed_loop_finite_response_h1          |         4 |       0.498 |       1.000 |    0.537 |             0.070 |                 0.498 |  0.800 |
| closed_loop_finite_response_h1          |         8 |       0.511 |       1.008 |    0.491 |             0.120 |                 0.511 |  0.900 |
| closed_loop_finite_response_h1_h2_h4_h8 |         1 |       0.688 |       0.888 |    0.652 |             0.380 |                 0.619 |  0.600 |
| closed_loop_finite_response_h1_h2_h4_h8 |         2 |       0.593 |       0.905 |    0.492 |             0.080 |                 0.533 |  0.800 |
| closed_loop_finite_response_h1_h2_h4_h8 |         4 |       0.483 |       0.896 |    0.504 |             0.100 |                 0.435 |  0.800 |
| closed_loop_finite_response_h1_h2_h4_h8 |         8 |       0.524 |       0.855 |    0.488 |             0.080 |                 0.471 |  0.600 |
| moving_tangent_interpolated_oracle      |         1 |       0.827 |       0.874 |    0.818 |             0.500 |                 0.827 |  0.900 |
| moving_tangent_interpolated_oracle      |         2 |       0.725 |       0.903 |    0.590 |             0.290 |                 0.725 |  0.600 |
| moving_tangent_interpolated_oracle      |         4 |       0.599 |       0.931 |    0.580 |             0.160 |                 0.599 |  0.778 |
| moving_tangent_interpolated_oracle      |         8 |       0.497 |       0.992 |    0.475 |             0.090 |                 0.497 |  0.600 |
| static_local_causal_oracle              |         1 |       0.792 |       0.823 |    0.801 |             0.500 |                 0.792 |  0.800 |
| static_local_causal_oracle              |         2 |       0.708 |       0.901 |    0.625 |             0.330 |                 0.708 |  0.700 |
| static_local_causal_oracle              |         4 |       0.579 |       0.934 |    0.485 |             0.080 |                 0.579 |  0.778 |
| static_local_causal_oracle              |         8 |       0.540 |       0.998 |    0.500 |             0.070 |                 0.540 |  0.800 |

All-horizon frozen gate pass by method: `{'closed_loop_finite_response_h1': False, 'closed_loop_finite_response_h1_h2_h4_h8': False, 'moving_tangent_interpolated_oracle': False, 'static_local_causal_oracle': False}`. Continuous semantic fidelity is reported separately and does not replace the legacy gate. This is a development experiment, not V14-D independent confirmation. Differences versus V13 are exploratory because the controller's finite-response derivative was not validated as a local Jacobian.

The all-horizon objective did not consistently improve h4/h8 over h1-only (h4 J direction fell from about 0.498 to 0.483; h8 rose only from about 0.511 to 0.524). It did not resolve long-horizon rotation or semantic divergence. h16 was not authorized.

Per-step predicted/actual improvement, trust ratio, control norm, residual, and atlas index are in `results/v14/processed/closed_loop_development_v14.parquet`.
