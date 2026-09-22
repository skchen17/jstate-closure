# Cross-State Coordinate Transport — V30

Five canonical source states were selected using only prewrite eligibility before V30 causal responses; this resource amendment changed only the source map and preserved all 280 frozen target–pair keys. Each source/target local map used ≥65 common TOKEN_TRAIN pairs, ridge α=1 or orthogonal Procrustes; target held-out token write never fit the map. The source's held-out contrast was projected into its TRAIN basis, mapped to the target's TRAIN basis, written into target-native REC+Conv cache, and judged against the target-native donor future. Raw source delta is **not** the primary test.

| family | canonical source | TRAIN pairs | source r95 | basis hash prefix |
|---|---|---|---|---|
| boolean_logic | v13-train-eef9accb656affcadbe4 | 72 | 54 | dcc5ddc5a73b11bc |
| modular_arithmetic | v13-train-b0c1c613e293661abb0b | 72 | 54 | 9901591dd29845e5 |
| short_graph_traversal | v13-train-4dad09829d6e18eb5f14 | 72 | 55 | c8a3be21588d451f |
| simple_state_transition | v13-train-551f383b519f280c1a95 | 72 | 55 | 8b953a135ecd8674 |
| variable_binding | v13-train-9ebb7221b0cd1647e246 | 72 | 54 | 57ff7d5f6a691e62 |

| condition | n | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| LOCAL_ORACLE_k64 | 60 | 0.807 | 0.882 | 0.597 |
| TRANSPORTED_LOCAL_RIDGE_k64 | 60 | 0.807 | 0.882 | 0.600 |
| TRANSPORTED_LOCAL_PROCRUSTES_k64 | 60 | 0.803 | 0.876 | 0.600 |

| condition | n | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| LOCAL_ORACLE_k64 | 120 | 0.834 | 0.845 | 0.556 |
| TRANSPORTED_LOCAL_RIDGE_k64 | 120 | 0.830 | 0.841 | 0.562 |
| TRANSPORTED_LOCAL_PROCRUSTES_k64 | 120 | 0.829 | 0.839 | 0.564 |

Neither map passes unseen-token causal gates. Similarity to the failing local ceiling is not shared-language confirmation. Transport hashes are per-row in `local_transport_*_v30.parquet`.
