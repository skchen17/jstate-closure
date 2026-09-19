# Finite causal reachability — V14 development only

Teacher-target residuals were measured for the same ten V13 validation cases used by the prior development oracle. The controller never receives a raw target cache; raw teacher state is used only to evaluate the teacher response label. Residual ratio is `||Y*−Y(P_k)|| / ||Y*−Y(P_0)||` for the selected horizon objective.

|   acceptance | method                                  |   raw_ratio |   step |   weighted_ratio |
|-------------:|:----------------------------------------|------------:|-------:|-----------------:|
|        1.000 | closed_loop_finite_response_h1          |       0.736 |      1 |            0.750 |
|        0.900 | closed_loop_finite_response_h1          |       0.637 |      2 |            0.655 |
|        0.556 | closed_loop_finite_response_h1          |       0.564 |      3 |            0.564 |
|        0.600 | closed_loop_finite_response_h1          |       0.492 |      4 |            0.579 |
|        0.900 | closed_loop_finite_response_h1_h2_h4_h8 |       0.762 |      1 |            0.871 |
|        0.556 | closed_loop_finite_response_h1_h2_h4_h8 |       0.677 |      2 |            0.845 |
|        0.600 | closed_loop_finite_response_h1_h2_h4_h8 |       0.546 |      3 |            0.777 |
|        0.667 | closed_loop_finite_response_h1_h2_h4_h8 |       0.484 |      4 |            0.724 |

Terminal case categories (development diagnostics, not independent reachability claims):

| method                                  | family                  | class                      |   cases |
|:----------------------------------------|:------------------------|:---------------------------|--------:|
| closed_loop_finite_response_h1          | boolean_logic           | partial_reduction_or_stall |       2 |
| closed_loop_finite_response_h1          | modular_arithmetic      | partial_reduction_or_stall |       2 |
| closed_loop_finite_response_h1          | short_graph_traversal   | intermediate_reduction     |       2 |
| closed_loop_finite_response_h1          | simple_state_transition | partial_reduction_or_stall |       2 |
| closed_loop_finite_response_h1          | variable_binding        | partial_reduction_or_stall |       2 |
| closed_loop_finite_response_h1_h2_h4_h8 | boolean_logic           | partial_reduction_or_stall |       2 |
| closed_loop_finite_response_h1_h2_h4_h8 | modular_arithmetic      | partial_reduction_or_stall |       2 |
| closed_loop_finite_response_h1_h2_h4_h8 | short_graph_traversal   | intermediate_reduction     |       1 |
| closed_loop_finite_response_h1_h2_h4_h8 | short_graph_traversal   | no_meaningful_reduction    |       1 |
| closed_loop_finite_response_h1_h2_h4_h8 | simple_state_transition | intermediate_reduction     |       2 |
| closed_loop_finite_response_h1_h2_h4_h8 | variable_binding        | partial_reduction_or_stall |       2 |

Backtracking enforces improvement in the weighted objective, not necessarily every unweighted target or horizon. A failure here cannot establish V14-E while the local finite-response model and independent confirmatory panel remain unvalidated.

Machine records: `results/v14/processed/closed_loop_development_v14.parquet`.
