# V39 — Falcon Higher Order Replication

Falcon development (80 states): second-order relative error median 0.5537 (state-bootstrap 95% CI [0.5010218089548233, 0.5826778245970267]), three-way fraction 0.5537, corrected signed projection coefficient p 0.0718, cosine 0.1399, sign stability 0.9125. Higher-order gate passes in 5/5 families; Q234 gate passes in 4/5. There are 0 below-floor states. This is a development replication, **not** V39-A qualification: validation was not opened after the mediator screen failed. Large f with small p is not signed amplification.

| Family | Falcon f | Falcon sign stability | Falcon Q234 fraction | Falcon Q234 cosine | Qwen f |
|---|---:|---:|---:|---:|---:|
| boolean_logic | 0.6038 | 1.0000 | 0.7056 | 0.8707 | 0.1784 |
| modular_arithmetic | 0.5527 | 0.9375 | 0.9123 | 0.8970 | 0.1536 |
| short_graph_traversal | 0.4132 | 0.8750 | 0.6917 | 0.8286 | 0.2218 |
| simple_state_transition | 0.4875 | 0.7500 | 0.7976 | 0.8533 | 0.1985 |
| variable_binding | 0.6257 | 1.0000 | 0.5765 | 0.7070 | 0.2063 |
