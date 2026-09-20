# V18 crossed state × action bank

Shared eight train-calibrated finite directions, two signs, primary α=0.5; the 400/80-state horizon panels additionally test α=1.0. Unreliable rows are retained with their status.

| role | h | α | states | rows | reliable | fraction |
|---|---|---|---|---|---|---|
| train | 1 | 0.5 | 2000 | 32000 | 32000 | 1.0000 |
| train | 1 | 1.0 | 400 | 6400 | 6400 | 1.0000 |
| train | 2 | 0.5 | 400 | 6400 | 6400 | 1.0000 |
| train | 2 | 1.0 | 400 | 6400 | 6400 | 1.0000 |
| train | 4 | 0.5 | 400 | 6400 | 6400 | 1.0000 |
| train | 4 | 1.0 | 400 | 6400 | 6400 | 1.0000 |
| train | 8 | 0.5 | 400 | 6400 | 6400 | 1.0000 |
| train | 8 | 1.0 | 400 | 6400 | 6400 | 1.0000 |
| validation | 1 | 0.5 | 400 | 6400 | 6400 | 1.0000 |
| validation | 1 | 1.0 | 80 | 1280 | 1280 | 1.0000 |
| validation | 2 | 0.5 | 80 | 1280 | 1280 | 1.0000 |
| validation | 2 | 1.0 | 80 | 1280 | 1280 | 1.0000 |
| validation | 4 | 0.5 | 80 | 1280 | 1280 | 1.0000 |
| validation | 4 | 1.0 | 80 | 1280 | 1280 | 1.0000 |
| validation | 8 | 0.5 | 80 | 1280 | 1280 | 1.0000 |
| validation | 8 | 1.0 | 80 | 1280 | 1280 | 1.0000 |

Family balance and reliable response coverage:

| role | family | states | rows | reliable fraction |
|---|---|---|---|---|
| train | boolean_logic | 400 | 15360 | 1.0000 |
| train | modular_arithmetic | 400 | 15360 | 1.0000 |
| train | short_graph_traversal | 400 | 15360 | 1.0000 |
| train | simple_state_transition | 400 | 15360 | 1.0000 |
| train | variable_binding | 400 | 15360 | 1.0000 |
| validation | boolean_logic | 80 | 3072 | 1.0000 |
| validation | modular_arithmetic | 80 | 3072 | 1.0000 |
| validation | short_graph_traversal | 80 | 3072 | 1.0000 |
| validation | simple_state_transition | 80 | 3072 | 1.0000 |
| validation | variable_binding | 80 | 3072 | 1.0000 |

Train/validation state IDs are disjoint and split SHA256 values are `2861b138b83a195b9f18e2f8587cfab982713c2b651b379c0eecaaabf90b48f4` / `a18fffe4b5640b1d530a6bee37b73610a4ac07dad309cfe4aeb1e998c7505fe4`.
Action coordinate indices: [5, 1, 7, 20, 11, 19, 17, 2]. Validation responses did not select actions.

Pre-action current J and raw REC/Conv/KV come from V13 clean layer-23 records; action-response J is read at layer 30.
A state-layer correction archived 401 pilot metadata records; median/max live-to-frozen J relative L2 drift was 0.0175/0.0314. The frozen V13 J is the canonical current-state value. Action responses were unchanged.
Fresh V18 clean-greedy teacher tokens match the shared frozen V13 prefix in 2390/2400 states and match the h1 token in 2391/2400; later-token drift is recorded, not used as a filter.
The single-GPU runtime amendment retained identical h1 response stacks for all 16 actions in a read-only equivalence check.
