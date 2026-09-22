# Contextual REC Correction — V32

For each of 10 development and 10 validation targets, target Conv and recipient KV are fixed. Matched REC is compared with recipient, wrong-token, same-family wrong-state, cross-family wrong-state, shuffled, sign-flipped and random same-norm REC. All variants use the same six probes.

| role | REC condition | median relative donor L2 |
|---|---|---|
| development | CROSS_FAMILY_WRONG_STATE_REC | 0.547 |
| development | MATCHED_REC | 0.161 |
| development | RANDOM_SAME_NORM_REC | 1.292 |
| development | RECIPIENT_REC | 0.321 |
| development | SAME_FAMILY_WRONG_STATE_REC | 0.351 |
| development | SHUFFLED_REC | 2.419 |
| development | SIGN_FLIPPED_REC | 0.522 |
| development | WRONG_TOKEN_REC | 0.318 |
| validation | CROSS_FAMILY_WRONG_STATE_REC | 0.522 |
| validation | MATCHED_REC | 0.202 |
| validation | RANDOM_SAME_NORM_REC | 1.488 |
| validation | RECIPIENT_REC | 0.287 |
| validation | SAME_FAMILY_WRONG_STATE_REC | 0.367 |
| validation | SHUFFLED_REC | 2.329 |
| validation | SIGN_FLIPPED_REC | 0.446 |
| validation | WRONG_TOKEN_REC | 0.306 |

Matched REC beats all four primary off-context controls in development 1.000 and validation 1.000 of targets. Cross-state REC is off-manifold; its failure alone does not identify state-specific gating. The mapping was frozen before context-specific responses but after factorial observations, so V32-C is **not** promoted to a fully preregistered formal outcome. Sources: `context_mapping_*_v32.json`, `context_*_v32.parquet`.
