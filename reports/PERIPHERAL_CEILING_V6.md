# Peripheral Ceiling v6

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Mode: `strong multi-endpoint peripheral reference`
- Date: `2026-09-15`
- Verification Status: `repository-grounded; machine results cited below`
- Protocol: `peripheral_foundations_protocol_v6`
- causal/null freeze digest: `f62c55d21e5e220cd21ea81c19a28bf1d5249a60bdc35a08e3eb029dd61ec12e`
- strong ceiling freeze digest: `6d4c77a3e33edf9e977e04293b73b77d153a15af091b1c496fe869e92ec23170`
- compact freeze digest: `07b51492b188b23da2ad66708cd3520e7b568a65bfd867ec019da7d301c24417`
- delivery/report freeze digest: `160d746a5b9b14e226acf876989e3196f971f321dbb9d2fb3720f7da68319840`


## Result

- Selected J-only/history baseline: `j_only_residual`
- Selected full-peripheral model: `full_remainder_gated`
- Strong peripheral ceiling authorized: **True**
- Authorization gates: `{"causal_direction": true, "causal_projection": true, "global_profile": false, "semantic": false}`
- J-only parameters: `5533201`; full-R parameters: `7369233`
- Absolute causal-direction cosine: J-only `0.013485`, full-R `0.469736`
- Absolute rollout-test next-J cosine: J-only `0.936841`, full-R `0.952951`
- Absolute semantic accuracy: J-only `0.635069`, full-R `0.627546`

| endpoint                         | full-minus-baseline              |
|:---------------------------------|:---------------------------------|
| action_correct                   | -0.031250 [-0.056296, -0.006854] |
| action_cross_entropy             | -0.433120 [-0.571387, -0.310876] |
| causal_projection_rmse           | 0.002717 [0.002231, 0.003196]    |
| next_j_cosine                    | 0.015335 [0.013135, 0.017540]    |
| top_causal_dimension_rmse        | 0.000467 [0.000343, 0.000586]    |
| causal::causal_direction_cosine  | 0.456251 [0.379585, 0.538002]    |
| causal::output_sign_agreement    | 0.060606 [-0.151515, 0.262626]   |
| causal::semantic_delta_agreement | -0.171717 [-0.292929, -0.060606] |

The validation-only architecture rule selected the gated full-R model; causal-test scores were excluded from model selection. The strong ceiling is authorized by causal-direction and causal-projection gates, not by the global-profile gate (semantic accuracy lost **3.12** percentage points) and not by output-sign/semantic-delta agreement. This is evidence for small-energy, causally predictive peripheral information, not a uniformly better behavioral predictor.

The models jointly optimize next-J profile, causal directions/top coordinates, and semantic action. Causal-sensitive directions were fitted only on `causal_fit`; all reported causal gains use disjoint `causal_test` pairs. The full operational remainder is a train-fitted residual representation, not a proof about every possible non-J coordinate system. Attention training reported a nondeterministic CUDA backward kernel; three frozen seeds are retained in the estimates.

## Validation-only architecture screen

| architecture             |   parameter_count |   validation_next_j_cosine |   validation_action_accuracy |   validation_causal_projection_rmse |   selection_score |
|:-------------------------|------------------:|---------------------------:|-----------------------------:|------------------------------------:|------------------:|
| j_only_residual          |           5533201 |                   0.937767 |                     0.608772 |                           0.0150147 |          0.948441 |
| j_history_attention      |          10527761 |                   0.934453 |                     0.571053 |                           0.0153777 |          0.944336 |
| full_remainder_linear    |           5530641 |                   0.934656 |                     0.570175 |                           0.0154829 |          0.944511 |
| full_remainder_gated     |           7369233 |                   0.952837 |                     0.613158 |                           0.0127524 |          0.963825 |
| full_remainder_residual  |           6844433 |                   0.943955 |                     0.607895 |                           0.0132955 |          0.954783 |
| full_remainder_attention |          11838993 |                   0.952012 |                     0.607895 |                           0.0119831 |          0.962972 |

## Family-wise gains

| family                  | next-J cosine gain              | semantic accuracy gain           | causal-direction gain         |
|:------------------------|:--------------------------------|:---------------------------------|:------------------------------|
| boolean_logic           | 0.009764 [0.008052, 0.011534]   | 0.005729 [-0.051569, 0.060677]   | 0.418142 [0.265779, 0.598650] |
| modular_arithmetic      | 0.030440 [0.024340, 0.036791]   | -0.084280 [-0.135732, -0.034714] | 0.424324 [0.265252, 0.587363] |
| short_graph_traversal   | 0.014348 [0.010434, 0.018439]   | -0.064323 [-0.139329, 0.013027]  | 0.572635 [0.510563, 0.636580] |
| simple_state_transition | 0.024922 [0.022004, 0.027630]   | -0.025000 [-0.052611, 0.004688]  | 0.526628 [0.396408, 0.713122] |
| variable_binding        | -0.000155 [-0.002719, 0.002584] | 0.002344 [-0.041406, 0.049219]   | 0.412822 [0.262214, 0.587617] |

Figure: `results/v6/figures/peripheral_ceiling_endpoints_v6.png`.
