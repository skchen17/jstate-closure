# Action Data Scaling — V22

| strategy | actions | S1 cross-state L2 | local S2 action-ceiling L2 | local cosine |
|---|---:|---:|---:|---:|
| causal_d_optimal | 12 | 1.0016 | 0.9951 | 0.0558 |
| causal_d_optimal | 24 | 1.0123 | 0.9954 | 0.2747 |
| causal_d_optimal | 48 | 1.0310 | 0.9958 | 0.4576 |
| causal_d_optimal | 96 | 1.4368 | 1.4571 | 0.6195 |
| causal_d_optimal | 128 | 1.3226 | 1.5474 | 0.6206 |
| random_architecture_balanced | 12 | 1.1079 | 1.0184 | 0.5871 |
| random_architecture_balanced | 24 | 1.1434 | 1.0089 | 0.6077 |
| random_architecture_balanced | 48 | 1.1511 | 1.0029 | 0.6191 |
| random_architecture_balanced | 96 | 1.0393 | 1.0170 | 0.5951 |
| random_architecture_balanced | 128 | 1.0366 | 1.0074 | 0.5945 |
| raw_geometry_maximin | 12 | 1.0089 | 0.9272 | 0.5606 |
| raw_geometry_maximin | 24 | 1.0075 | 0.9236 | 0.5719 |
| raw_geometry_maximin | 48 | 1.0080 | 0.9209 | 0.5738 |
| raw_geometry_maximin | 96 | 1.0107 | 0.8921 | 0.5851 |
| raw_geometry_maximin | 128 | 1.0366 | 1.0074 | 0.5945 |

Saturation by strategy: `{"causal_d_optimal": false, "random_architecture_balanced": true, "raw_geometry_maximin": false}`. Overall saturation is not established; `ACTION_DATA_LIMITED=TRUE`. The adverse 128-action point is reported, not hidden.
