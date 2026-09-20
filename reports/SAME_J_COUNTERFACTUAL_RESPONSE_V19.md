# V19 boundary-held-J finite action response

Primary estimand M=[Y(Pq,a)-Y(Pq,0)]−[Y(P0,a)-Y(P0,0)]. R0=Y01−Y00; Rq=Y11−Y10. Only matched-realized-action and reliably realized q rows enter the primary summary; failed rows remain in the raw denominator and are labeled.

## Validation target × horizon

| h | target | states | median ‖N‖ | median ‖R0‖ | median ‖Rq‖ | median ‖M‖ | median M/R | M/R 95% CI | cos(R0,Rq) | ‖Rq‖/‖R0‖ |
|---|---|---|---|---|---|---|---|---|---|---|
| h1 | j | 100 | 1.5629 | 1.6802 | 1.4065 | 0.8805 | 0.5087 | [0.5028, 0.5222] | 0.8945 | 0.9636 |
| h1 | logits | 100 | 3.3232 | 3.4594 | 3.3282 | 2.1785 | 0.5432 | [0.5288, 0.5606] | 0.8634 | 0.9894 |
| h1 | semantic_continuous | 100 | 3.1043 | 3.6985 | 3.3848 | 2.1102 | 0.4894 | [0.4665, 0.5082] | 0.8902 | 0.9775 |
| h1 | workspace | 100 | 3.4119 | 3.1002 | 3.0948 | 1.7728 | 0.5022 | [0.4895, 0.5159] | 0.8915 | 0.9985 |
| h1 | stacked_normalized | 100 | 6.2626 | 6.5562 | 6.5003 | 4.2223 | 0.5642 | [0.5521, 0.5760] | 0.8453 | 0.9920 |
| h2 | j | 15 | 0.4783 | 0.2726 | 0.2828 | 0.1451 | 0.6458 | [0.6393, 0.6661] | 0.7846 | 1.0039 |
| h2 | logits | 15 | 1.6748 | 1.1815 | 1.1464 | 0.5292 | 0.5941 | [0.5716, 0.6079] | 0.8190 | 1.0038 |
| h2 | semantic_continuous | 15 | 2.0508 | 1.3928 | 1.3573 | 0.5839 | 0.5194 | [0.5015, 0.5742] | 0.8710 | 0.9932 |
| h2 | workspace | 15 | 1.5761 | 0.6092 | 0.7192 | 0.2437 | 0.6583 | [0.6389, 0.6955] | 0.7815 | 1.0387 |
| h2 | stacked_normalized | 15 | 3.4762 | 2.1440 | 2.1910 | 0.8849 | 0.5915 | [0.5797, 0.6172] | 0.8110 | 1.0032 |
| h4 | j | 15 | 0.3240 | 0.2000 | 0.2114 | 0.0808 | 0.7707 | [0.7639, 0.8197] | 0.6871 | 1.0175 |
| h4 | logits | 15 | 1.0845 | 0.7189 | 0.7508 | 0.3401 | 0.7414 | [0.7056, 0.7634] | 0.6999 | 1.0255 |
| h4 | semantic_continuous | 15 | 1.2207 | 0.7763 | 0.7985 | 0.5407 | 0.7190 | [0.6433, 0.7748] | 0.7508 | 1.0191 |
| h4 | workspace | 15 | 0.3920 | 0.2183 | 0.2571 | 0.1063 | 0.6951 | [0.6629, 0.7200] | 0.7381 | 1.0224 |
| h4 | stacked_normalized | 15 | 1.8380 | 1.1508 | 1.2445 | 0.7043 | 0.7125 | [0.6503, 0.7541] | 0.7282 | 1.0201 |
| h8 | j | 15 | 0.1082 | 0.0410 | 0.0463 | 0.0356 | 0.8807 | [0.8502, 0.9201] | 0.5249 | 1.0573 |
| h8 | logits | 15 | 0.4430 | 0.1921 | 0.2124 | 0.1944 | 0.9615 | [0.9204, 0.9930] | 0.4314 | 1.0478 |
| h8 | semantic_continuous | 15 | 0.4636 | 0.2376 | 0.2955 | 0.2707 | 0.9395 | [0.8549, 0.9728] | 0.4399 | 1.1346 |
| h8 | workspace | 15 | 0.1641 | 0.1035 | 0.1109 | 0.1086 | 0.9631 | [0.9328, 0.9965] | 0.3601 | 1.0638 |
| h8 | stacked_normalized | 15 | 0.6865 | 0.3324 | 0.3812 | 0.3551 | 0.9595 | [0.8958, 1.0002] | 0.4057 | 1.1005 |

Norms in the table use frozen V16 target-block normalization. Physical raw-block norms for J, logits, continuous semantic score and workspace, plus normalized ratios and response direction cosines, are retained per row in `results/v19/processed/factorial_metrics_validation_v19.parquet`.

## Task families

| group | states | median M/R | 95% state-bootstrap CI | median N/R | match rate |
|---|---|---|---|---|---|
| boolean_logic | 20 | 0.5030 | [0.4791, 0.5547] | 0.8511 | 1.0000 |
| modular_arithmetic | 20 | 0.5369 | [0.5172, 0.5521] | 0.9545 | 1.0000 |
| short_graph_traversal | 20 | 0.5933 | [0.5780, 0.6232] | 0.7565 | 1.0000 |
| simple_state_transition | 20 | 0.5921 | [0.5755, 0.6085] | 0.8014 | 1.0000 |
| variable_binding | 20 | 0.5601 | [0.5375, 0.5907] | 0.8534 | 1.0000 |

## Probe actions

| group | states | median M/R | 95% state-bootstrap CI | median N/R | match rate |
|---|---|---|---|---|---|
| 1 | 100 | 0.5881 | [0.5769, 0.5985] | 1.0003 | 1.0000 |
| 11 | 15 | 0.7054 | [0.5961, 0.8522] | 1.8833 | 1.0000 |
| 17 | 15 | 0.4714 | [0.4416, 0.4859] | 1.7713 | 1.0000 |
| 19 | 15 | 0.6111 | [0.5934, 0.6511] | 5.0173 | 1.0000 |
| 2 | 15 | 0.4019 | [0.3348, 0.5049] | 2.2846 | 1.0000 |
| 20 | 100 | 0.5010 | [0.4819, 0.5207] | 0.7142 | 1.0000 |
| 5 | 100 | 0.6108 | [0.5954, 0.6206] | 0.7888 | 1.0000 |
| 7 | 100 | 0.6194 | [0.5981, 0.6422] | 0.9836 | 1.0000 |

## Primary task families (train-frozen active q only, descriptive)

| group | states | rows | median M/R | median N/R0 |
|---|---|---|---|---|
| boolean_logic | 20 | 920 | 0.7108 | 1.2334 |
| modular_arithmetic | 20 | 920 | 0.7097 | 1.5189 |
| short_graph_traversal | 20 | 920 | 0.8027 | 1.2952 |
| simple_state_transition | 20 | 920 | 0.8331 | 1.4040 |
| variable_binding | 20 | 920 | 0.8002 | 1.4178 |

## Primary probe actions (train-frozen active q only, descriptive)

| group | states | rows | median M/R | median N/R0 |
|---|---|---|---|---|
| 1 | 100 | 1000 | 0.8576 | 1.4643 |
| 2 | 15 | 150 | 0.5525 | 3.1397 |
| 5 | 100 | 1000 | 0.8539 | 1.1597 |
| 7 | 100 | 1000 | 0.7906 | 1.3804 |
| 11 | 15 | 150 | 0.8833 | 2.5674 |
| 17 | 15 | 150 | 0.6302 | 2.6589 |
| 19 | 15 | 150 | 0.6949 | 7.2238 |
| 20 | 100 | 1000 | 0.6689 | 1.0331 |

Formal primary active-q matched result: M/R 0.7899, state-bootstrap CI [0.7577, 0.8097], 100 independent validation base states and 4600 correlated response rows. Equivalence margin 0.10; dependence lower floor 0.02 were frozen before any factorial response.
