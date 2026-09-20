# V18 strong h1 state-context ceiling

All conditions use a unified state × action predictor and an equal 128-dimensional state feature budget. M0–M4 are fit on train states, with train-only nested hyperparameter selection; validation is held out.

| model | context | parameters | J rel-L2 | stack rel-L2 |
|---|---|---|---|---|
| M0_additive_linear | action_only | 39456 | 0.5620 | 0.6732 |
| M1_bilinear | action_only | 334368 | 0.5620 | 0.6732 |
| M2_quadratic_tensor | action_only | 631584 | 0.5334 | 0.6040 |
| M3_low_rank_interaction | action_only | 53024 | 0.5600 | 0.7688 |
| M4_small_state_conditioned_mlp | action_only | 71200 | 0.4810 | 0.5450 |
| M0_additive_linear | j | 39456 | 0.5499 | 0.6557 |
| M1_bilinear | j | 334368 | 0.4084 | 0.5079 |
| M2_quadratic_tensor | j | 631584 | 0.3294 | 0.3706 |
| M3_low_rank_interaction | j | 53024 | 0.4143 | 0.6304 |
| M4_small_state_conditioned_mlp | j | 71200 | 0.2740 | 0.2879 |
| M0_additive_linear | j_rec | 39456 | 0.5501 | 0.6557 |
| M1_bilinear | j_rec | 334368 | 0.4075 | 0.5069 |
| M2_quadratic_tensor | j_rec | 631584 | 0.3280 | 0.3687 |
| M3_low_rank_interaction | j_rec | 53024 | 0.4144 | 0.6300 |
| M4_small_state_conditioned_mlp | j_rec | 71200 | 0.2717 | 0.2861 |
| M0_additive_linear | j_conv | 39456 | 0.5494 | 0.6554 |
| M1_bilinear | j_conv | 334368 | 0.4062 | 0.5058 |
| M2_quadratic_tensor | j_conv | 631584 | 0.3261 | 0.3670 |
| M3_low_rank_interaction | j_conv | 53024 | 0.4132 | 0.6289 |
| M4_small_state_conditioned_mlp | j_conv | 71200 | 0.2743 | 0.2855 |
| M0_additive_linear | j_kv | 39456 | 0.5506 | 0.6565 |
| M1_bilinear | j_kv | 334368 | 0.4118 | 0.5124 |
| M2_quadratic_tensor | j_kv | 631584 | 0.3345 | 0.3781 |
| M3_low_rank_interaction | j_kv | 53024 | 0.4176 | 0.6353 |
| M4_small_state_conditioned_mlp | j_kv | 71200 | 0.2757 | 0.2957 |
| M0_additive_linear | j_rec_conv | 39456 | 0.5498 | 0.6555 |
| M1_bilinear | j_rec_conv | 334368 | 0.4068 | 0.5065 |
| M2_quadratic_tensor | j_rec_conv | 631584 | 0.3271 | 0.3680 |
| M3_low_rank_interaction | j_rec_conv | 53024 | 0.4139 | 0.6296 |
| M4_small_state_conditioned_mlp | j_rec_conv | 71200 | 0.2739 | 0.2861 |
| M0_additive_linear | j_rec_kv | 39456 | 0.5504 | 0.6561 |
| M1_bilinear | j_rec_kv | 334368 | 0.4095 | 0.5096 |
| M2_quadratic_tensor | j_rec_kv | 631584 | 0.3311 | 0.3732 |
| M3_low_rank_interaction | j_rec_kv | 53024 | 0.4155 | 0.6320 |
| M4_small_state_conditioned_mlp | j_rec_kv | 71200 | 0.2750 | 0.2902 |
| M0_additive_linear | j_conv_kv | 39456 | 0.5500 | 0.6558 |
| M1_bilinear | j_conv_kv | 334368 | 0.4086 | 0.5083 |
| M2_quadratic_tensor | j_conv_kv | 631584 | 0.3297 | 0.3712 |
| M3_low_rank_interaction | j_conv_kv | 53024 | 0.4149 | 0.6310 |
| M4_small_state_conditioned_mlp | j_conv_kv | 71200 | 0.2762 | 0.2914 |
| M0_additive_linear | j_rec_conv_kv | 39456 | 0.5501 | 0.6558 |
| M1_bilinear | j_rec_conv_kv | 334368 | 0.4083 | 0.5079 |
| M2_quadratic_tensor | j_rec_conv_kv | 631584 | 0.3292 | 0.3703 |
| M3_low_rank_interaction | j_rec_conv_kv | 53024 | 0.4143 | 0.6312 |
| M4_small_state_conditioned_mlp | j_rec_conv_kv | 71200 | 0.2732 | 0.2890 |
| M0_additive_linear | full_raw | 39456 | 0.5503 | 0.6561 |
| M1_bilinear | full_raw | 334368 | 0.4090 | 0.5093 |
| M2_quadratic_tensor | full_raw | 631584 | 0.3304 | 0.3730 |
| M3_low_rank_interaction | full_raw | 53024 | 0.4162 | 0.6326 |
| M4_small_state_conditioned_mlp | full_raw | 71200 | 0.2738 | 0.2910 |

Primary model target-level validation metrics:

| target | context | rel-L2 | direction median | magnitude ratio | sign agreement |
|---|---|---|---|---|---|
| j | j | 0.2740 | 0.9607 | 0.9809 | 0.8927 |
| j | j_rec_conv_kv | 0.2732 | 0.9606 | 0.9771 | 0.8927 |
| logits | j | 0.3075 | 0.9495 | 0.9579 | 0.8877 |
| logits | j_rec_conv_kv | 0.3084 | 0.9488 | 0.9593 | 0.8858 |
| semantic_continuous | j | 0.2873 | 0.9566 | 0.9709 | 0.8728 |
| semantic_continuous | j_rec_conv_kv | 0.2892 | 0.9564 | 0.9724 | 0.8728 |
| workspace | j | 0.2652 | 0.9514 | 0.9783 | 0.8543 |
| workspace | j_rec_conv_kv | 0.2661 | 0.9513 | 0.9818 | 0.8543 |
| stacked_normalized | j | 0.2879 | 0.9543 | 0.9700 | 0.8802 |
| stacked_normalized | j_rec_conv_kv | 0.2890 | 0.9545 | 0.9706 | 0.8799 |

Formal primary model: `M4_small_state_conditioned_mlp`. Full clean REC+Conv+KV over J gives absolute J/stack relative-L2 gains **0.0008/-0.0011**; material gate 0.05 passed: **False**.
Paired state bootstrap 95% CI (J/stack): [-0.0023345811102397375, 0.003882224497058244] / [-0.00304272113224257, 0.0007479536486805177].
Family stack gains: {'boolean_logic': -0.0031779583599853, 'modular_arithmetic': -0.002836229528851264, 'short_graph_traversal': -0.0024720414851274186, 'simple_state_transition': 0.0017036131160380052, 'variable_binding': 0.0009343358408681612}.
V17 reported J/stack gains 0.0126/0.0113 on its smaller linear-kernel reused-action cohort. V18 changes cohort size, shared-action crossing and model class, so the comparison is directional rather than a paired effect estimate.
A small gain does not prove conditional independence or a complete Markov state.
