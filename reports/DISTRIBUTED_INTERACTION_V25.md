# Distributed Causal Interaction — V25

Strict factorial estimand: `Y00=clean; Y10=clean+B; Y01=clean+C; Y11=clean+(B+C)`. All `160` corrected rows are append-only; the earlier persistent-FULL interpretation was preserved but superseded for the formal interaction decision.

| metric | value |
|---|---:|
| broad interaction ratio median | 0.340985 |
| bootstrap 2.5% lower bound | 0.319602 |
| additive relative L2 median | 0.340985 |
| interaction cosine median | -0.013222 |
| centered interaction r95 | 77 |

| family | median ratio |
|---|---:|
| boolean_logic | 0.280331 |
| modular_arithmetic | 0.360449 |
| short_graph_traversal | 0.332038 |
| simple_state_transition | 0.342103 |
| variable_binding | 0.348545 |

The writeback numerical gate failed because the minimum B-branch cosine was `0.633638`. Consequently `V25-A = FALSE` conservatively, despite a nonzero interaction statistic.
