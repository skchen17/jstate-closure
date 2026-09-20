# V20 unseen scale, pair and dense actions

The pre-frozen 10-base, 40-operator-state diagnostic panel measures 2× amplitude on one train and one validation direction, two unseen validation-direction pairs, and a five-direction dense mixture. The model sees each new continuous action descriptor but was never trained on the exact diagnostic combination. Action writeback and P0/Pq matching remain audited.

| action | type | stack relative L2 | median cosine | reliable rate | matched rate |
|---|---|---|---|---|---|
| unseen_dense_0_1_2_20_21 | unseen_dense | 0.8257 | 0.7127 | 1.0000 | 1.0000 |
| unseen_pair_20_16 | unseen_pair | 0.8487 | 0.7138 | 1.0000 | 1.0000 |
| unseen_pair_21_17 | unseen_pair | 1.0011 | 0.1153 | 1.0000 | 1.0000 |
| unseen_scale_train_0 | unseen_amplitude_train | 0.8354 | 0.9578 | 1.0000 | 1.0000 |
| unseen_scale_validation_20 | unseen_amplitude_validation | 0.8901 | 0.6671 | 1.0000 | 1.0000 |

Measured joint action is compared with the model's `G(C,a+b)`; additionally, additivity residual `||R(a+b)-R(a)-R(b)||/||R(a+b)||` is `{"unseen_pair_20_16": {"relative_additivity_residual_median": 0.28480471670627594}, "unseen_pair_21_17": {"relative_additivity_residual_median": 0.18583906441926956}}`. Additivity is not assumed. The joint sign/scale/composition gate is **False**. Raw diagnostics: `results/v20/processed/operator_action_diagnostics_v20.parquet`.
