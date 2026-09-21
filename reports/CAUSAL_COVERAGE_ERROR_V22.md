# Causal Coverage and Error — V22

| metric family | quantity | Spearman | Pearson |
|---|---|---:|---:|
| Z1 | leverage | -0.1452 | 0.0107 |
| Z1 | nearest_abs_cosine | -0.2258 | -0.0854 |
| Z1 | span_residual | 0.1452 | -0.0451 |
| Z2 | leverage | -0.2456 | 0.0056 |
| Z2 | nearest_abs_cosine | -0.2819 | -0.0639 |
| Z2 | span_residual | 0.2435 | -0.0643 |
| Z6 | leverage | -0.0696 | -0.0205 |
| Z6 | nearest_abs_cosine | 0.0484 | 0.0601 |
| Z6 | span_residual | -0.1309 | -0.2090 |

Raw and architecture span residuals decrease with action count, while causal Z6 reaches near-zero projected residual by 96 directions. Held-out error remains weakly associated with these distances, so causal projected coverage alone is not sufficient.
