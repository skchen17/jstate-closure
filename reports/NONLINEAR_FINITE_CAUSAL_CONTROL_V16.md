# Nonlinear finite causal control — V16

**Not launched by the frozen gate.** No M0–M4 candidate passed the held-out J and stacked finite-response gate. Nonlinear MPC requires that gate before development; a controller fit to an unvalidated response map would repeat V15's unsupported-actuator error. No V16 h1/h2/h4/h8 causal-control metrics, trust ratios or predicted-versus-actual improvement exist. Sequence MPC and independent confirmation were not opened.

Historical comparisons remain frozen: h1 V15 actuator-aware development direction nan, magnitude 0.000, output nan; V13 static/moving and V14 development results are preserved in their own records. Raw teacher self-reference is a label ceiling only (identity=1), never a target-cache bypass. A direct V16-vs-historical controller comparison cannot be made because no V16 controller was authorized. No historical full causal gate was removed: direction ≥0.8, magnitude 0.8–1.2, output ≥0.8, legacy semantic ≥0.8 and sign ≥0.8 at h1/h2/h4/h8 remain mandatory for any future independent control claim.

Frozen same-panel historical development metrics (not V16 results):

| method                                  |   horizon |   direction |   magnitude |   output |   semantic_continuous |   semantic_legacy |   sign |
|:----------------------------------------|----------:|------------:|------------:|---------:|----------------------:|------------------:|-------:|
| actuator_calibrated_h1                  |         1 |       0.290 |       0.307 |    0.308 |                 0.232 |             0.230 |  0.400 |
| actuator_calibrated_h1                  |         2 |       0.303 |       0.496 |    0.317 |                 0.243 |             0.020 |  0.600 |
| actuator_calibrated_h1                  |         4 |       0.453 |       0.706 |    0.472 |                 0.363 |             0.010 |  0.500 |
| actuator_calibrated_h1                  |         8 |       0.508 |       0.777 |    0.543 |                 0.406 |             0.070 |  0.400 |
| actuator_calibrated_h1_h2_h4            |         1 |       0.126 |       0.234 |    0.301 |                 0.075 |             0.020 |  0.200 |
| actuator_calibrated_h1_h2_h4            |         2 |       0.257 |       0.413 |    0.327 |                 0.154 |             0.040 |  0.400 |
| actuator_calibrated_h1_h2_h4            |         4 |       0.436 |       0.555 |    0.517 |                 0.261 |             0.020 |  0.500 |
| actuator_calibrated_h1_h2_h4            |         8 |       0.528 |       0.565 |    0.541 |                 0.317 |             0.050 |  0.500 |
| actuator_calibrated_h1_h2_h4_h8         |         1 |       0.190 |       0.218 |    0.308 |                 0.114 |             0.070 |  0.200 |
| actuator_calibrated_h1_h2_h4_h8         |         2 |       0.276 |       0.383 |    0.309 |                 0.166 |             0.030 |  0.600 |
| actuator_calibrated_h1_h2_h4_h8         |         4 |       0.465 |       0.542 |    0.488 |                 0.279 |             0.030 |  0.500 |
| actuator_calibrated_h1_h2_h4_h8         |         8 |       0.526 |       0.548 |    0.610 |                 0.316 |             0.040 |  0.500 |
| closed_loop_finite_response_h1          |         1 |       0.699 |       0.981 |    0.671 |                 0.699 |             0.430 |  0.800 |
| closed_loop_finite_response_h1          |         2 |       0.535 |       1.005 |    0.514 |                 0.535 |             0.110 |  0.800 |
| closed_loop_finite_response_h1          |         4 |       0.498 |       1.000 |    0.537 |                 0.498 |             0.070 |  0.800 |
| closed_loop_finite_response_h1          |         8 |       0.511 |       1.008 |    0.491 |                 0.511 |             0.120 |  0.900 |
| closed_loop_finite_response_h1_h2_h4_h8 |         1 |       0.688 |       0.888 |    0.652 |                 0.619 |             0.380 |  0.600 |
| closed_loop_finite_response_h1_h2_h4_h8 |         2 |       0.593 |       0.905 |    0.492 |                 0.533 |             0.080 |  0.800 |
| closed_loop_finite_response_h1_h2_h4_h8 |         4 |       0.483 |       0.896 |    0.504 |                 0.435 |             0.100 |  0.800 |
| closed_loop_finite_response_h1_h2_h4_h8 |         8 |       0.524 |       0.855 |    0.488 |                 0.471 |             0.080 |  0.600 |
| moving_tangent_interpolated_oracle      |         1 |       0.827 |       0.874 |    0.818 |                 0.827 |             0.500 |  0.900 |
| moving_tangent_interpolated_oracle      |         2 |       0.725 |       0.903 |    0.590 |                 0.725 |             0.290 |  0.600 |
| moving_tangent_interpolated_oracle      |         4 |       0.599 |       0.931 |    0.580 |                 0.599 |             0.160 |  0.778 |
| moving_tangent_interpolated_oracle      |         8 |       0.497 |       0.992 |    0.475 |                 0.497 |             0.090 |  0.600 |
| raw_teacher_label_self_comparison       |         1 |       1.000 |       1.000 |    1.000 |                 1.000 |             1.000 |  1.000 |
| raw_teacher_label_self_comparison       |         2 |       1.000 |       1.000 |    1.000 |                 1.000 |             1.000 |  1.000 |
| raw_teacher_label_self_comparison       |         4 |       1.000 |       1.000 |    1.000 |                 1.000 |             1.000 |  1.000 |
| raw_teacher_label_self_comparison       |         8 |       1.000 |       1.000 |    1.000 |                 1.000 |             1.000 |  1.000 |
| static_local_causal_oracle              |         1 |       0.792 |       0.823 |    0.801 |                 0.792 |             0.500 |  0.800 |
| static_local_causal_oracle              |         2 |       0.708 |       0.901 |    0.625 |                 0.708 |             0.330 |  0.700 |
| static_local_causal_oracle              |         4 |       0.579 |       0.934 |    0.485 |                 0.579 |             0.080 |  0.778 |
| static_local_causal_oracle              |         8 |       0.540 |       0.998 |    0.500 |                 0.540 |             0.070 |  0.800 |
