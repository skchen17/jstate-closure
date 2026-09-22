# Global Write Basis — V30

The fixed 13,369,344-component REC+Conv write vector was fit on 90 development states × 8 frozen TOKEN_TRAIN contrasts = 720 rows. These are centered **affine** PCA approximations (mean + Uc), not exact write dimensions or model-state dimensions. All basis tensors remain off-repository scratch; their SHA-256 hashes are committed.

| basis | TRAIN rows | rank | r90 | r95 | r99 | basis hash prefix |
|---|---|---|---|---|---|---|
| FAMILY_boolean_logic | 144 | 143 | 48 | 59 | 79 | c899f2859477d4c7 |
| FAMILY_modular_arithmetic | 144 | 143 | 48 | 58 | 75 | 49804befb388af32 |
| FAMILY_short_graph_traversal | 144 | 143 | 50 | 60 | 77 | 506681d1be8f8527 |
| FAMILY_simple_state_transition | 144 | 143 | 51 | 61 | 79 | ef80da22f9f86a16 |
| FAMILY_variable_binding | 144 | 143 | 49 | 59 | 78 | 422ae32a24131c6d |
| GLOBAL | 720 | 719 | 74 | 131 | 320 | 371aa65ff0f79261 |
| LOFO_boolean_logic | 576 | 575 | 69 | 112 | 256 | bea60272a1f2e188 |
| LOFO_modular_arithmetic | 576 | 575 | 70 | 113 | 260 | 3ae9e51aef69f4a3 |
| LOFO_short_graph_traversal | 576 | 575 | 72 | 122 | 279 | bcd5cecd0d500377 |
| LOFO_simple_state_transition | 576 | 575 | 72 | 122 | 277 | 16d7c07ba5c7a46a |
| LOFO_variable_binding | 576 | 575 | 70 | 113 | 257 | d97e5e843d0cf2e5 |

The global r95=131 already exceeds k64. Geometry alone is not the verdict: the held-out causal OOD results below decide fidelity.
