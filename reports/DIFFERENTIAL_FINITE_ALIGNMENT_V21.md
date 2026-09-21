# V21 differential–finite alignment on identical P0/Pq states

Validation Spearman correlation between per-pair JVP rotation and finite rotation is **0.5669** (family-stratified base bootstrap 95% CI **[0.1541, 0.8287]**); Pearson r **0.5632**. Same-state P0/Pq JVP–finite median dominant-subspace overlaps are **0.8550/0.8679**, and P0 response-column median cosine is **0.8205**.

| family | validation within-family Spearman rotation correlation |
|---|---:|
| boolean_logic | 0.4000 |
| modular_arithmetic | 0.9000 |
| short_graph_traversal | 0.7000 |
| simple_state_transition | 0.1000 |
| variable_binding | 0.9000 |

Strong shared-geometry support under the corrected frozen **all five families positive** gate is **True**. Similar ranks alone are not treated as operator equality. The half/full finite-action scale and BF16 quantization remain potential sources of differential–finite divergence. Detailed per-base angles and overlaps: `paired_geometry_analysis_v21.parquet`; bootstrap: `paired_geometry_bootstrap_v21.json`.
