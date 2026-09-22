# Leave-One-Family-Out Generalization — V30

For each family, the LOFO basis excludes **all** TRAIN rows of that family. The table reports JOINT-OOD relative L2 separately; no failed family is averaged away.

| role | held-out family | exact REC+Conv | Conv | global32 | global64 | LOFO64 | local oracle64 | transport ridge64 |
|---|---|---|---|---|---|---|---|---|
| development | boolean_logic | 0.307 | 0.395 | 0.612 | 0.586 | 0.600 | 0.595 | 0.604 |
| development | modular_arithmetic | 0.189 | 0.297 | 0.558 | 0.548 | 0.579 | 0.527 | 0.531 |
| development | short_graph_traversal | 0.273 | 0.361 | 0.718 | 0.697 | 0.724 | 0.676 | 0.674 |
| development | simple_state_transition | 0.240 | 0.316 | 0.611 | 0.536 | 0.555 | 0.530 | 0.526 |
| development | variable_binding | 0.314 | 0.421 | 0.673 | 0.651 | 0.664 | 0.648 | 0.659 |
| validation | boolean_logic | 0.247 | 0.324 | 0.573 | 0.539 | 0.581 | 0.540 | 0.542 |
| validation | modular_arithmetic | 0.156 | 0.304 | 0.576 | 0.564 | 0.578 | 0.562 | 0.565 |
| validation | short_graph_traversal | 0.255 | 0.312 | 0.612 | 0.565 | 0.609 | 0.519 | 0.559 |
| validation | simple_state_transition | 0.234 | 0.319 | 0.617 | 0.599 | 0.595 | 0.570 | 0.585 |
| validation | variable_binding | 0.203 | 0.314 | 0.599 | 0.573 | 0.618 | 0.564 | 0.574 |

No k≤64 family-OOD causal representation meets the frozen ≥4/5-family rule in both development and validation. The exact natural ceiling remains distinct from compact generalization.
