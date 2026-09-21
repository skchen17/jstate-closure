# Layerwise Causal Convergence — V25

An append-only correction assigns exact-zero response matrices rank zero. Layers 0–23 have exact-zero observed responses; layer 24 is the first nonzero layer. Pooled centered r95 rises from `56` at layer 24 to `99` at layer 31, a reduction fraction of `-0.767857` (negative means rank expansion). Absolute cosine falls from `0.283520` to `0.159618`.

| family | r95 at layer 24 | r95 at layer 31 |
|---|---:|---:|
| boolean_logic | 32 | 56 |
| modular_arithmetic | 19 | 31 |
| short_graph_traversal | 21 | 62 |
| simple_state_transition | 18 | 59 |
| variable_binding | 35 | 50 |

`MANY_TO_ONE_CAUSAL_CONVERGENCE = FALSE`; `INTERNAL_CAUSAL_CONVERGENCE_SUPPORTED = FALSE`.
