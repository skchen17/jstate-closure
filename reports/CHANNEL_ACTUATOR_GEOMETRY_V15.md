# Channel-resolved actuator geometry — V15

`R_REC`, `R_CONV`, `R_KV` and `R_joint` were measured at the same ε, state and probe directions; K/V also have separate raw writeback transfer curves. Target-block and stacked spectra are in `finite_operator_spectra_v15.json`.

Stacked-normalized rank at 64 probes across five train anchors:

| base_trial_id                  | channel   |   rank_90 |   rank_95 |   rank_99 |   stable_rank |
|:-------------------------------|:----------|----------:|----------:|----------:|--------------:|
| v13-train-90d330b478659e3d0bc3 | joint     |         6 |         8 |        13 |         2.218 |
| v13-train-90d330b478659e3d0bc3 | recurrent |         4 |         4 |         7 |         2.065 |
| v13-train-90d330b478659e3d0bc3 | conv      |         4 |         5 |         8 |         1.948 |
| v13-train-90d330b478659e3d0bc3 | kv        |         2 |         3 |         5 |         1.245 |
| v13-train-6bb01e7e53b688215c6e | joint     |         7 |        10 |        16 |         2.805 |
| v13-train-6bb01e7e53b688215c6e | recurrent |         3 |         4 |         8 |         2.181 |
| v13-train-6bb01e7e53b688215c6e | conv      |         4 |         5 |         9 |         2.029 |
| v13-train-6bb01e7e53b688215c6e | kv        |         3 |         4 |         6 |         1.589 |
| v13-train-4e63d69f5c3f40704813 | joint     |         7 |         9 |        14 |         3.024 |
| v13-train-4e63d69f5c3f40704813 | recurrent |         3 |         4 |         8 |         1.730 |
| v13-train-4e63d69f5c3f40704813 | conv      |         4 |         6 |        10 |         2.292 |
| v13-train-4e63d69f5c3f40704813 | kv        |         2 |         2 |         3 |         1.146 |
| v13-train-c6eec494888d6d518e28 | joint     |         7 |        10 |        15 |         3.508 |
| v13-train-c6eec494888d6d518e28 | recurrent |         3 |         5 |         8 |         2.146 |
| v13-train-c6eec494888d6d518e28 | conv      |         4 |         6 |        10 |         2.526 |
| v13-train-c6eec494888d6d518e28 | kv        |         2 |         2 |         3 |         1.205 |
| v13-train-07385890263e531ff8e3 | joint     |         7 |         9 |        14 |         2.381 |
| v13-train-07385890263e531ff8e3 | recurrent |         4 |         5 |         8 |         2.212 |
| v13-train-07385890263e531ff8e3 | conv      |         4 |         5 |         8 |         1.783 |
| v13-train-07385890263e531ff8e3 | kv        |         2 |         2 |         4 |         1.891 |

SNR-qualified column fractions: `{'conv': 0.4401041666666667, 'joint': 0.546875, 'kv': 0.01171875, 'recurrent': 0.1640625}`. Median `||R_joint-(R_REC+R_CONV+R_KV)||/||R_joint||` by target: `{'j': 0.10560545330015618, 'logits': 0.1958444811671171, 'semantic_continuous': 0.20340082998229436, 'workspace': 0.297095397911219}`. Interaction is finite-response nonadditivity, not a literal independent-channel decomposition. V14's Conv-removal ablation is preserved; V15 distinguishes its raw actuator dead-zone (larger than REC at small scales) from its actual causal response. Per-direction channel norms and interaction residuals: `channel_interaction_v15.parquet`.
The data support measuring joint REC/Conv/KV effects, but do not prove all three channels are intrinsically necessary: V14's restricted ablation found material Conv removal and no material KV removal under its criterion.
