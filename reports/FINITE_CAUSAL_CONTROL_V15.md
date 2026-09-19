# Actuator-calibrated development control — V15

The compact basis is a realized-cost-weighted SVD of **finite response columns**, not PCA or raw autograd singular vectors. The denominator is a diagonal approximation using each column's measured realized-state norm; raw-state cross-Gram and nonlinear combination writeback are not assumed away. The basis coefficients and held-out-probe target-coverage comparison with an autograd-JVP target basis are machine-readable. **The finite basis covers less held-out-probe target energy than the autograd basis in the tested 512-probe anchor**; no basis superiority is claimed.

|   rank |   r95_weighted |   finite_basis_heldout_target_coverage |   autograd_basis_heldout_finite_target_coverage |
|-------:|---------------:|---------------------------------------:|------------------------------------------------:|
|  4.000 |         22.000 |                                  0.283 |                                           0.628 |
|  8.000 |         22.000 |                                  0.428 |                                           0.727 |
| 16.000 |         22.000 |                                  0.664 |                                           0.842 |
| 22.000 |         22.000 |                                  0.763 |                                           0.872 |
| 32.000 |         22.000 |                                  0.849 |                                           0.907 |
| 64.000 |         22.000 |                                  0.964 |                                           0.982 |

On the same ten V13 validation cases, the V15 controller used actual ε=1 finite responses, train/frozen output scales, h1 / h1+h2+h4 / h1+h2+h4+h8 objectives, at most four steps, rank `min(8, local r95)`, and realized-response trust decisions. Every step records requested control norm, realized state norm, channel gains, prediction error, actual residual, acceptance and trust ratio. A teacher raw cache was used **only** to generate the teacher response label, never as candidate input or basis. These coordinates are numerical actuator controls, not cognitive or biological state variables.

Median residual curves:

| method                          |   step |   raw_ratio |   weighted_ratio |   acceptance |
|:--------------------------------|-------:|------------:|-----------------:|-------------:|
| actuator_calibrated_h1          |      1 |       0.937 |            0.933 |        0.800 |
| actuator_calibrated_h1          |      2 |       0.806 |            0.826 |        0.750 |
| actuator_calibrated_h1          |      3 |       0.806 |            0.826 |        0.167 |
| actuator_calibrated_h1          |      4 |       0.850 |            0.878 |        1.000 |
| actuator_calibrated_h1_h2_h4    |      1 |       0.955 |            0.978 |        0.600 |
| actuator_calibrated_h1_h2_h4    |      2 |       0.829 |            0.845 |        0.833 |
| actuator_calibrated_h1_h2_h4    |      3 |       0.738 |            0.834 |        0.600 |
| actuator_calibrated_h1_h2_h4    |      4 |       0.667 |            0.801 |        0.333 |
| actuator_calibrated_h1_h2_h4_h8 |      1 |       0.919 |            0.990 |        0.600 |
| actuator_calibrated_h1_h2_h4_h8 |      2 |       0.751 |            0.893 |        0.833 |
| actuator_calibrated_h1_h2_h4_h8 |      3 |       0.748 |            0.897 |        0.400 |
| actuator_calibrated_h1_h2_h4_h8 |      4 |       0.655 |            0.829 |        1.000 |

Those stepwise medians have different surviving case sets and need not be monotone even when a within-case accepted step improves the weighted objective. The ten-case terminal analysis includes stopped cases:

| method                          |   case_count |   terminal_raw_ratio |   terminal_weighted_ratio |   raw_monotone_fraction |   weighted_monotone_fraction |   terminal_improvement_fraction |
|:--------------------------------|-------------:|---------------------:|--------------------------:|------------------------:|-----------------------------:|--------------------------------:|
| actuator_calibrated_h1          |           10 |                0.854 |                     0.863 |                   0.900 |                        1.000 |                           0.800 |
| actuator_calibrated_h1_h2_h4    |           10 |                0.899 |                     0.902 |                   0.800 |                        1.000 |                           0.600 |
| actuator_calibrated_h1_h2_h4_h8 |           10 |                0.835 |                     0.911 |                   0.800 |                        1.000 |                           0.600 |

Weighted residual is monotone by acceptance construction, but raw teacher residual was not monotone for every case and terminal improvement was not universal.

Same development-panel controller comparison (raw teacher is the target label, so its self-fidelity is trivially 1, not a competing learned controller):

| method                                  |   horizon |   direction |   magnitude |   output |   semantic_legacy |   semantic_continuous |   sign |
|:----------------------------------------|----------:|------------:|------------:|---------:|------------------:|----------------------:|-------:|
| actuator_calibrated_h1                  |         1 |       0.290 |       0.307 |    0.308 |             0.230 |                 0.232 |  0.400 |
| actuator_calibrated_h1                  |         2 |       0.303 |       0.496 |    0.317 |             0.020 |                 0.243 |  0.600 |
| actuator_calibrated_h1                  |         4 |       0.453 |       0.706 |    0.472 |             0.010 |                 0.363 |  0.500 |
| actuator_calibrated_h1                  |         8 |       0.508 |       0.777 |    0.543 |             0.070 |                 0.406 |  0.400 |
| actuator_calibrated_h1_h2_h4            |         1 |       0.126 |       0.234 |    0.301 |             0.020 |                 0.075 |  0.200 |
| actuator_calibrated_h1_h2_h4            |         2 |       0.257 |       0.413 |    0.327 |             0.040 |                 0.154 |  0.400 |
| actuator_calibrated_h1_h2_h4            |         4 |       0.436 |       0.555 |    0.517 |             0.020 |                 0.261 |  0.500 |
| actuator_calibrated_h1_h2_h4            |         8 |       0.528 |       0.565 |    0.541 |             0.050 |                 0.317 |  0.500 |
| actuator_calibrated_h1_h2_h4_h8         |         1 |       0.190 |       0.218 |    0.308 |             0.070 |                 0.114 |  0.200 |
| actuator_calibrated_h1_h2_h4_h8         |         2 |       0.276 |       0.383 |    0.309 |             0.030 |                 0.166 |  0.600 |
| actuator_calibrated_h1_h2_h4_h8         |         4 |       0.465 |       0.542 |    0.488 |             0.030 |                 0.279 |  0.500 |
| actuator_calibrated_h1_h2_h4_h8         |         8 |       0.526 |       0.548 |    0.610 |             0.040 |                 0.316 |  0.500 |
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
| raw_teacher_label_self_comparison       |         1 |       1.000 |       1.000 |    1.000 |             1.000 |                 1.000 |  1.000 |
| raw_teacher_label_self_comparison       |         2 |       1.000 |       1.000 |    1.000 |             1.000 |                 1.000 |  1.000 |
| raw_teacher_label_self_comparison       |         4 |       1.000 |       1.000 |    1.000 |             1.000 |                 1.000 |  1.000 |
| raw_teacher_label_self_comparison       |         8 |       1.000 |       1.000 |    1.000 |             1.000 |                 1.000 |  1.000 |
| static_local_causal_oracle              |         1 |       0.792 |       0.823 |    0.801 |             0.500 |                 0.792 |  0.800 |
| static_local_causal_oracle              |         2 |       0.708 |       0.901 |    0.625 |             0.330 |                 0.708 |  0.700 |
| static_local_causal_oracle              |         4 |       0.579 |       0.934 |    0.485 |             0.080 |                 0.579 |  0.778 |
| static_local_causal_oracle              |         8 |       0.540 |       0.998 |    0.500 |             0.070 |                 0.540 |  0.800 |

Residual decrease is **not** causal fidelity pass. The frozen finite-response linearity gate failed, so these results are exploratory development only; no independent V15 finalist was eligible. Historical legacy top-k semantic is reported alongside continuous semantic and has not been replaced.
