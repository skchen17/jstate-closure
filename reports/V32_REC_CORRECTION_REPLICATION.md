# REC Correction Replication — V32

Exact native factorials used new 100 development and 50 validation states, one primary ordinary natural token fork per state, six prewrite probes, and recipient-native KV. The independent 50-state final remains sealed. Relative improvement is `(||Ydonor−Y01||−||Ydonor−Y11||)/||Ydonor−Y01||`; state-bootstrap lower bounds use 2,000 resamples.

| role | n | positive fraction | median reduction | bootstrap LB | median alignment | alignment LB | families | replication gate | alignment gate |
|---|---|---|---|---|---|---|---|---|---|
| development | 100 | 1.000 | 0.410 | 0.346 | 0.809 | 0.759 | 5 | True | True |
| validation | 50 | 1.000 | 0.372 | 0.309 | 0.779 | 0.723 | 5 | True | True |

| role | family | n | positive fraction | median reduction | median alignment |
|---|---|---|---|---|---|
| development | boolean_logic | 20 | 1.000 | 0.372 | 0.780 |
| development | modular_arithmetic | 20 | 1.000 | 0.520 | 0.880 |
| development | short_graph_traversal | 20 | 1.000 | 0.365 | 0.777 |
| development | simple_state_transition | 20 | 1.000 | 0.431 | 0.824 |
| development | variable_binding | 20 | 1.000 | 0.324 | 0.739 |
| validation | boolean_logic | 10 | 1.000 | 0.355 | 0.759 |
| validation | modular_arithmetic | 10 | 1.000 | 0.433 | 0.824 |
| validation | short_graph_traversal | 10 | 1.000 | 0.341 | 0.752 |
| validation | simple_state_transition | 10 | 1.000 | 0.409 | 0.807 |
| validation | variable_binding | 10 | 1.000 | 0.362 | 0.770 |

| role | frozen token category | n | median reduction | median alignment | positive fraction |
|---|---|---|---|---|---|
| development | CJK_SURFACE_UNRESOLVED | 18 | 0.677 | 0.947 | 1.000 |
| development | FUNCTION_WORD | 19 | 0.196 | 0.599 | 1.000 |
| development | LEXICAL_SURFACE | 23 | 0.480 | 0.854 | 1.000 |
| development | NUMERIC | 25 | 0.384 | 0.803 | 1.000 |
| development | PUNCTUATION_OR_STRUCTURE | 15 | 0.365 | 0.774 | 1.000 |
| validation | CJK_SURFACE_UNRESOLVED | 11 | 0.650 | 0.937 | 1.000 |
| validation | FUNCTION_WORD | 11 | 0.260 | 0.673 | 1.000 |
| validation | LEXICAL_SURFACE | 10 | 0.356 | 0.764 | 1.000 |
| validation | NUMERIC | 12 | 0.320 | 0.738 | 1.000 |
| validation | PUNCTUATION_OR_STRUCTURE | 6 | 0.344 | 0.754 | 1.000 |

REC-only remains weak: relative donor L2 medians development/validation 0.998/0.997; Conv-only is 0.283/0.288, joint is 0.163/0.180. This establishes a conditional causal *effect of exact REC field replacement* on the tested donor-fidelity endpoint, not a localized mechanism or REC as an independent carrier. Sources: `factorial_*_v32.parquet`, matching vectors and audits.
