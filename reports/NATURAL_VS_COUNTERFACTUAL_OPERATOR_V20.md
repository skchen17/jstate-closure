# V20 natural versus counterfactual operator manifolds

The distributions are separated: clean P0 states versus same-boundary-J Pq states. Centered equal-base action-response fingerprints yield r95 and entropy effective rank:

| role | distribution / q | states | r95 | entropy rank |
|---|---|---|---|---|
| operator_train | P0 | 150 | 39 | 11.8346 |
| operator_train | conv_causal1 | 150 | 44 | 11.9003 |
| operator_train | joint_causal11 | 150 | 39 | 13.0970 |
| operator_train | pooled_natural_plus_counterfactual | 600 | 50 | 16.0742 |
| operator_train | rec_arch4_amended | 150 | 35 | 11.6390 |
| operator_validation | P0 | 50 | 20 | 8.2223 |
| operator_validation | conv_causal1 | 50 | 23 | 9.1584 |
| operator_validation | joint_causal11 | 50 | 20 | 9.1144 |
| operator_validation | pooled_natural_plus_counterfactual | 200 | 34 | 14.0395 |
| operator_validation | rec_arch4_amended | 50 | 19 | 7.7261 |

This is an empirical rank on the frozen 18-direction local finite-action panel. It neither proves a global manifold dimension nor implies physical cache compression. Natural J→oracle-coordinate validation R²=0.1394; natural relative L2=0.9044; counterfactual relative L2=1.1608. The J predictor was fitted on natural training states only.
