# V17 action-response context localization

These are **finite-action response** gains, not historical next-J residual localization. In the matched-capacity ceiling, REC contributes the largest individual measured gain, Conv a smaller gain, KV little. The joint context is **not** better than REC alone, so there is no evidence here for a necessary joint-channel mechanism. All gains are below the material raw-ceiling threshold.

The separate fixed-base conditional test cross-fits train target residuals and raw-feature residuals by state, then applies a frozen 32-feature correction on validation. Negative gain means the correction worsened prediction. It is not a conditional-independence proof.

| raw channel(s) | ceiling J gain | ceiling stack gain | fixed-base conditional J gain | fixed-base conditional stack gain | stack bootstrap 95% CI |
|---|---:|---:|---:|---:|---|
| rec | 0.0160 | 0.0115 | -0.0026 | -0.0083 | `[-0.020889075657980573, 0.0038278152127249375]` |
| conv | 0.0124 | 0.0094 | -0.0006 | -0.0020 | `[-0.008026579344350781, 0.004764448848052189]` |
| kv | 0.0040 | 0.0028 | -0.0264 | -0.0122 | `[-0.03415690636615126, 0.008502011192684705]` |
| rec+conv | 0.0150 | 0.0110 | 0.0006 | -0.0032 | `[-0.012802850233398819, 0.007821474267631712]` |
| rec+kv | 0.0110 | 0.0093 | -0.0157 | -0.0072 | `[-0.0281494018261882, 0.01568069449835887]` |
| conv+kv | 0.0094 | 0.0089 | -0.0149 | -0.0053 | `[-0.023565569579662177, 0.015509761949787369]` |
| rec+conv+kv | 0.0126 | 0.0113 | -0.0104 | -0.0005 | `[-0.019534563660787544, 0.02037226311169412]` |

Source: `results/v17/processed/state_context_ceiling_v17.json`, `conditional_raw_residual_v17.json`, and both prediction Parquets. Conditional design freeze `1f7f47ea02310b6a7068688be4656a2fb968cd08dfa98f7c63021f94b8c7b57f`.
