# Empirical finite causal reachability — V15 development

The one-step columns are local finite-response basis directions. Multi-step columns concatenate the remeasured response maps after attempted writebacks in a common frozen target coordinate system; no full state transition Jacobian is claimed. Projection floors refer only to the tested rank-limited actuator span and weighted teacher residual, not global reachability.

| objective   |   one_step_rank |   multi_step_rank |   one_step_floor |   multi_step_floor |   outside_fraction |
|:------------|----------------:|------------------:|-----------------:|-------------------:|-------------------:|
| h1          |           7.000 |            18.000 |            0.845 |              0.712 |              1.000 |
| h1_h2_h4    |           6.500 |            17.000 |            0.938 |              0.853 |              1.000 |
| h1_h2_h4_h8 |           6.000 |            12.500 |            0.977 |              0.891 |              1.000 |

`TARGET_OUTSIDE_TESTED_ACTUATOR_REACHABLE_SPACE` is recorded per case only when the multi-step projection floor exceeds the predeclared 0.2 threshold. Since the finite linearity gate failed, this is a descriptive development diagnostic, not a formal proof of actuator impossibility. Machine rows: `finite_controllability_v15.parquet`.
