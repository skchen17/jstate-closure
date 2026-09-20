# Minimal nonlinear action dimension — V16

Frozen candidates: k=2,4,8,16,32, nested across all five direction proposals. The response gate requires median J **and** normalized-stack direction ≥0.90, relative L2 ≤0.30, norm ratio 0.8–1.2 on state, strict sign, strict scale, strict pair and unseen dense heldouts. No model/dimension met every required holdout: `NO_COMPACT_NONLINEAR_ACTION_DIMENSION_IDENTIFIED`. This means **within tested models/protocol only**, not mathematical nonexistence. Within-state, across-state and across-family `k_min` are not assigned when the gate fails. Family-exclusion rows are diagnostic, not an independent final bank.

Best observed model per dimension on held-out states (not sufficient alone to pass):

|   k | model               |   test_count |   j_direction |   j_relative_l2 |   stacked_normalized_direction |   stacked_normalized_relative_l2 | gate_pass   |
|----:|:--------------------|-------------:|--------------:|----------------:|-------------------------------:|---------------------------------:|:------------|
|   2 | piecewise_quadratic |          360 |         0.907 |           0.500 |                          0.908 |                            0.466 | False       |
|   4 | piecewise_quadratic |          901 |         0.928 |           0.463 |                          0.904 |                            0.490 | False       |
|   8 | piecewise_quadratic |         1213 |         0.929 |           0.453 |                          0.912 |                            0.467 | False       |
|  16 | piecewise_quadratic |         1881 |         0.922 |           0.464 |                          0.906 |                            0.477 | False       |
|  32 | piecewise_quadratic |         3327 |         0.922 |           0.456 |                          0.893 |                            0.489 | False       |

All M0 linear, M1 quadratic, M2 channel-bilinear, M3 train-selected sparse cubic and M4 clean-J-conditioned piecewise quadratic comparisons, including strict action-design holdouts and output/semantic metrics: `/data/CSK/J-space-project/jstate-closure/results/v16/processed/nonlinear_model_comparison_v16.parquet`. M5 small MLP was not launched because the frozen resource priority and failed simpler models did not justify its cost; V16-F therefore remains unclaimed.
