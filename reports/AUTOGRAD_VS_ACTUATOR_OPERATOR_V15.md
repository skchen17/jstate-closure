# Ideal autograd JVP versus actual finite actuator — V15

`Jv` is the frozen V13 exact-autograd differential column. `Rεv=[Y(W(P,+εv))-Y(W(P,-εv))]/(2ε)` uses real persistent-state writeback and the frozen J, selected-logit, semantic-continuous and workspace readouts. Target-block normalization uses only V13 **train** JVP columns: `{'j': 0.00637603926805087, 'logits': 0.37930603920841477, 'semantic_continuous': 0.4250155426398896, 'workspace': 0.14939923966331858}`. The fifth target is their dimension-normalized stack. No V13/V14 result was rewritten.

The first epsilon meeting the frozen selection rule was found for only **16/100** train state×direction×channel combinations; unselected trials remain in the Parquet rather than being dropped from the denominator. Selection requires both signed J effects ≥ `0.00680280`, ≥10× measured repeat noise, nonzero state writeback, state gain 0.2–5, and signed-effect norm ratio ≤5. This last check is only a coarse saturation screen, not a proof of local linearity.

Selected comparisons:

| channel   | target              |   count |   cosine |   relative_l2 |   norm_ratio |   realized_state_cosine |
|:----------|:--------------------|--------:|---------:|--------------:|-------------:|------------------------:|
| conv      | j                   |       6 |    0.903 |         0.446 |        1.017 |                   1.000 |
| conv      | logits              |       6 |    0.749 |         0.701 |        1.014 |                   1.000 |
| conv      | semantic_continuous |       6 |    0.748 |         0.692 |        1.057 |                   1.000 |
| conv      | stacked_normalized  |       6 |    0.784 |         0.651 |        1.015 |                   1.000 |
| conv      | workspace           |       6 |    0.717 |         0.753 |        0.953 |                   1.000 |
| joint     | j                   |       7 |    0.997 |         0.080 |        1.025 |                   1.000 |
| joint     | logits              |       7 |    0.984 |         0.194 |        1.037 |                   1.000 |
| joint     | semantic_continuous |       7 |    0.985 |         0.175 |        0.997 |                   1.000 |
| joint     | stacked_normalized  |       7 |    0.981 |         0.196 |        1.011 |                   1.000 |
| joint     | workspace           |       7 |    0.955 |         0.299 |        1.020 |                   1.000 |
| recurrent | j                   |       3 |    0.317 |         0.951 |        0.358 |                   1.000 |
| recurrent | logits              |       3 |    0.199 |         1.017 |        0.565 |                   1.000 |
| recurrent | semantic_continuous |       3 |    0.031 |         1.056 |        0.392 |                   1.000 |
| recurrent | stacked_normalized  |       3 |    0.130 |         1.036 |        0.431 |                   1.000 |
| recurrent | workspace           |       3 |    0.511 |         0.893 |        0.628 |                   1.000 |

Precision decomposition: construction/addition/storage/consumption/readout are explicit fields in `precision_modes_v15.parquet`. `native_fp32_add_bf16_writeback` and `cast_after_add` are the same canonical interface and produce identical outputs; native BF16 add quantizes the increment before addition. FP32 shadow REC/Conv and channel-specific FP32 modes alter diagnostic forward semantics, so they cannot replace the canonical result. Full FP32 KV model consumption is `UNSUPPORTED_BY_FROZEN_MODEL_DTYPE_CONTRACT`; no pretrained weight or query dtype was changed.

At ε=1 (selected diagnostic probes), precision comparisons:

| mode                           | target    | status                                     |   count |   median_cosine |   median_relative_l2 |   median_realized_state_cosine |
|:-------------------------------|:----------|:-------------------------------------------|--------:|----------------:|---------------------:|-------------------------------:|
| cast_after_add                 | j         | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.977 |                0.186 |                          0.998 |
| cast_after_add                 | logits    | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.895 |                0.554 |                          0.998 |
| cast_after_add                 | workspace | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.898 |                0.625 |                          0.998 |
| fp32_conv_only                 | j         | SUPPORTED_DIAGNOSTIC                       |      10 |           0.977 |                0.186 |                          1.000 |
| fp32_conv_only                 | logits    | SUPPORTED_DIAGNOSTIC                       |      10 |           0.895 |                0.554 |                          1.000 |
| fp32_conv_only                 | workspace | SUPPORTED_DIAGNOSTIC                       |      10 |           0.898 |                0.625 |                          1.000 |
| fp32_kv_only                   | j         | UNSUPPORTED_BY_FROZEN_MODEL_DTYPE_CONTRACT |      10 |         nan     |              nan     |                        nan     |
| fp32_kv_only                   | logits    | UNSUPPORTED_BY_FROZEN_MODEL_DTYPE_CONTRACT |      10 |         nan     |              nan     |                        nan     |
| fp32_kv_only                   | workspace | UNSUPPORTED_BY_FROZEN_MODEL_DTYPE_CONTRACT |      10 |         nan     |              nan     |                        nan     |
| fp32_rec_only                  | j         | SUPPORTED_DIAGNOSTIC                       |      10 |           0.981 |                0.205 |                          0.998 |
| fp32_rec_only                  | logits    | SUPPORTED_DIAGNOSTIC                       |      10 |           0.899 |                0.495 |                          0.998 |
| fp32_rec_only                  | workspace | SUPPORTED_DIAGNOSTIC                       |      10 |           0.880 |                0.603 |                          0.998 |
| fp32_shadow_rec_conv           | j         | SUPPORTED_DIAGNOSTIC                       |      10 |           0.981 |                0.205 |                          1.000 |
| fp32_shadow_rec_conv           | logits    | SUPPORTED_DIAGNOSTIC                       |      10 |           0.899 |                0.495 |                          1.000 |
| fp32_shadow_rec_conv           | workspace | SUPPORTED_DIAGNOSTIC                       |      10 |           0.880 |                0.603 |                          1.000 |
| native_bf16_add                | j         | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.977 |                0.186 |                          0.998 |
| native_bf16_add                | logits    | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.895 |                0.554 |                          0.998 |
| native_bf16_add                | workspace | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.898 |                0.625 |                          0.998 |
| native_fp32_add_bf16_writeback | j         | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.977 |                0.186 |                          0.998 |
| native_fp32_add_bf16_writeback | logits    | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.895 |                0.554 |                          0.998 |
| native_fp32_add_bf16_writeback | workspace | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.898 |                0.625 |                          0.998 |

At ε=1, canonical joint requested/realized state cosine is near 1 while logits/workspace output mismatch persists. This supports `AUTOGRAD_DIFFERENTIAL_OPERATOR_DOES_NOT_MATCH_FINITE_ACTUATOR_OPERATOR_IN_TESTED_REGIME`; simple BF16 rounding alone does not explain every output mismatch. The direction/target-specific adaptive rows and full signed projections, norm ratios, state realization and ε are in `autograd_vs_actuator_v15.parquet`. Repeatability max relative L2: `0.0`.
