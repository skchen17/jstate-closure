# V20 independent confirmation of V19-B

The new bank has **250 base states**, disjoint from V18/V19 designated roles. The boundary J is bitwise identical and persistent snapshots differ on every tested pair. This confirmation preceded V20 operator fitting.

Active q: median M/R **0.7862**, state-bootstrap 95% CI **[0.7782, 0.7931]**, 10000 matched rows; realized-action match 1.0000. Median N/R0 1.2420; N=8.6635, R0=6.9011, Rq=6.6857, M=5.9706. Response cosine=0.6493; ||Rq||/||R0||=0.9430.

| family | median M/R |
|---|---|
| boolean_logic | 0.7253 |
| modular_arithmetic | 0.7258 |
| short_graph_traversal | 0.8203 |
| simple_state_transition | 0.8400 |
| variable_binding | 0.8116 |

Formal result: **V19_B_INDEPENDENTLY_CONFIRMED**. This establishes persistent-state modulation of the tested finite responses, not compactness or dynamics. Raw rows: `results/v20/processed/independent_v19_confirmation_v20.parquet` (`db85777c95a1113774b280b21b4543b3fd53880b918b6c98c9d996746d25496b`).
