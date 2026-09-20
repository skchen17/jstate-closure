# V20 unseen action generalization

Every validation operator state supplies its coordinate using **train actions only**. Frozen action partition is independent of state partition. Best diagnostic model metrics:

| test | stack relative L2 | J relative L2 | stack median cosine | J median cosine | stack median norm ratio |
|---|---|---|---|---|---|
| seen direction/new state | 0.8236 | 0.8408 | 0.8790 | 0.8907 | 0.1459 |
| unseen direction | 0.9273 | 1.0132 | 0.3828 | 0.2985 | 0.1919 |
| unseen sign/train direction | 1.0056 | 0.9264 | 0.5130 | 0.6701 | 0.1451 |
| unseen direction+sign | 1.0378 | 1.0726 | 0.1195 | -0.0144 | 0.1941 |

Family-wise unseen-direction relative L2: `{"boolean_logic": 0.914719198995127, "modular_arithmetic": 0.9512948995503161, "short_graph_traversal": 0.9278593231971451, "simple_state_transition": 0.9380887443778032, "variable_binding": 0.9215754106345686}`. A new state with a seen action is not evidence of action generalization. Final six directions were not opened because no encoder finalist was eligible.
