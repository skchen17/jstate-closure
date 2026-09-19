# Causal tangent transport — V14

V13's ten train-anchor local matrices were aligned at rank 16 with explicit sign/index-invariant mappings. Same-prompt successive-position mean fidelity:

|   j_direction_cosine |   logits_direction_cosine | method             |   raw_direction_cosine |   semantic_continuous_direction_cosine |   workspace_direction_cosine |
|---------------------:|--------------------------:|:-------------------|-----------------------:|---------------------------------------:|-----------------------------:|
|                0.584 |                     0.311 | grassmann_geodesic |                  0.796 |                                  0.282 |                        0.536 |
|                0.832 |                     0.351 | j_response         |                  0.462 |                                  0.282 |                        0.431 |
|                0.584 |                     0.311 | procrustes         |                  0.796 |                                  0.282 |                        0.536 |
|                0.597 |                     0.325 | projector          |                  0.833 |                                  0.293 |                        0.549 |
|                0.048 |                    -0.001 | unaligned          |                  0.038 |                                  0.001 |                        0.046 |

Procrustes and minimal Grassmann-geodesic endpoint transport coincide mathematically for these full-rank principal-angle alignments; their identical numbers are not independent replications. The reported `raw_direction_cosine` is actually a **512-dimensional probe-coordinate** cosine, not a metric-calibrated full raw-state cosine because the frozen probe directions need not be orthonormal. J-response alignment improves J-effect direction but can sacrifice probe-coordinate/output/semantic alignment, demonstrating a target-specific gauge choice. Only common selected logit IDs were compared across states; the comparison does not assert global coordinate identity or finite steering success.

Machine records: `results/v14/processed/transport_pairs_v14.parquet` (`1900` rows).
