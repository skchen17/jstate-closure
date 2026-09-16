# Persistent Causal Dataset v8

Protocol: `structured_persistent_state_protocol_v8`; freeze: `1cbb255ddfd42438a18fc374daceebdf0f7ec6fe6f56c94a3f914654c1ab61ad`.

## Difficulty calibration

|   attempted | family                  |   final_answer_accuracy |   final_answer_correct |   full_trajectory_accuracy |   full_trajectory_correct |   horizon |   parseable |   parseable_rate |
|------------:|:------------------------|------------------------:|-----------------------:|---------------------------:|--------------------------:|----------:|------------:|-----------------:|
|          30 | boolean_logic           |                1        |                     30 |                   0.833333 |                        25 |         2 |          30 |                1 |
|          30 | boolean_logic           |                0.966667 |                     29 |                   0.766667 |                        23 |         3 |          30 |                1 |
|          30 | boolean_logic           |                1        |                     30 |                   0.666667 |                        20 |         5 |          30 |                1 |
|          30 | modular_arithmetic      |                0.766667 |                     23 |                   0.733333 |                        22 |         2 |          30 |                1 |
|          30 | modular_arithmetic      |                0.966667 |                     29 |                   0.933333 |                        28 |         3 |          30 |                1 |
|          30 | modular_arithmetic      |                0.633333 |                     19 |                   0.566667 |                        17 |         5 |          30 |                1 |
|          30 | short_graph_traversal   |                0.9      |                     27 |                   0.9      |                        27 |         2 |          30 |                1 |
|          30 | short_graph_traversal   |                0.933333 |                     28 |                   0.9      |                        27 |         3 |          30 |                1 |
|          30 | short_graph_traversal   |                0.866667 |                     26 |                   0.866667 |                        26 |         5 |          30 |                1 |
|          30 | simple_state_transition |                0.8      |                     24 |                   0.8      |                        24 |         2 |          30 |                1 |
|          30 | simple_state_transition |                0.9      |                     27 |                   0.9      |                        27 |         3 |          30 |                1 |
|          30 | simple_state_transition |                0.633333 |                     19 |                   0.633333 |                        19 |         5 |          30 |                1 |
|          30 | variable_binding        |                0.833333 |                     25 |                   0.833333 |                        25 |         2 |          30 |                1 |
|          30 | variable_binding        |                0.933333 |                     28 |                   0.766667 |                        23 |         3 |          30 |                1 |
|          30 | variable_binding        |                0.866667 |                     26 |                   0.633333 |                        19 |         5 |          30 |                1 |

## Preserved pre-freeze competence failure

The first formal horizon assignment was rejected before causal execution because modular-arithmetic horizon 3 fell below the 70% competence rule on final test.

|   attempted | family                  |   final_answer_accuracy |   final_answer_correct |   full_trajectory_accuracy |   full_trajectory_correct |   horizon |   parseable |   parseable_rate | split      |
|------------:|:------------------------|------------------------:|-----------------------:|---------------------------:|--------------------------:|----------:|------------:|-----------------:|:-----------|
|          80 | boolean_logic           |                0.95     |                     76 |                   0.9      |                        72 |         2 |          76 |         0.95     | final_test |
|          80 | modular_arithmetic      |                0.65     |                     52 |                   0.6375   |                        51 |         3 |          55 |         0.6875   | final_test |
|          80 | short_graph_traversal   |                0.95     |                     76 |                   0.9125   |                        73 |         5 |          77 |         0.9625   | final_test |
|          80 | simple_state_transition |                0.875    |                     70 |                   0.8      |                        64 |         3 |          68 |         0.85     | final_test |
|          80 | variable_binding        |                0.975    |                     78 |                   0.8625   |                        69 |         2 |          70 |         0.875    | final_test |
|         180 | boolean_logic           |                0.927778 |                    167 |                   0.838889 |                       151 |         2 |         167 |         0.927778 | train      |
|         180 | modular_arithmetic      |                0.688889 |                    124 |                   0.677778 |                       122 |         3 |         129 |         0.716667 | train      |
|         180 | short_graph_traversal   |                0.933333 |                    168 |                   0.911111 |                       164 |         5 |         176 |         0.977778 | train      |
|         180 | simple_state_transition |                0.9      |                    162 |                   0.8      |                       144 |         3 |         159 |         0.883333 | train      |
|         180 | variable_binding        |                0.961111 |                    173 |                   0.855556 |                       154 |         2 |         154 |         0.855556 | train      |
|          80 | boolean_logic           |                0.9      |                     72 |                   0.75     |                        60 |         2 |          69 |         0.8625   | validation |
|          80 | modular_arithmetic      |                0.7375   |                     59 |                   0.7125   |                        57 |         3 |          59 |         0.7375   | validation |
|          80 | short_graph_traversal   |                0.975    |                     78 |                   0.9125   |                        73 |         5 |          76 |         0.95     | validation |
|          80 | simple_state_transition |                0.8625   |                     69 |                   0.7875   |                        63 |         3 |          70 |         0.875    | validation |
|          80 | variable_binding        |                0.975    |                     78 |                   0.9      |                        72 |         2 |          73 |         0.9125   | validation |

## Frozen formal teacher competence

|   attempted | family                  |   final_answer_accuracy |   final_answer_correct |   full_trajectory_accuracy |   full_trajectory_correct |   horizon |   parseable |   parseable_rate | split      |
|------------:|:------------------------|------------------------:|-----------------------:|---------------------------:|--------------------------:|----------:|------------:|-----------------:|:-----------|
|          80 | boolean_logic           |                0.9625   |                     77 |                   0.7875   |                        63 |         2 |          80 |                1 | final_test |
|          80 | modular_arithmetic      |                0.7125   |                     57 |                   0.7125   |                        57 |         2 |          80 |                1 | final_test |
|          80 | short_graph_traversal   |                0.925    |                     74 |                   0.925    |                        74 |         5 |          80 |                1 | final_test |
|          80 | simple_state_transition |                0.7625   |                     61 |                   0.7375   |                        59 |         3 |          80 |                1 | final_test |
|          80 | variable_binding        |                0.8625   |                     69 |                   0.8625   |                        69 |         2 |          80 |                1 | final_test |
|         180 | boolean_logic           |                0.977778 |                    176 |                   0.794444 |                       143 |         2 |         180 |                1 | train      |
|         180 | modular_arithmetic      |                0.833333 |                    150 |                   0.822222 |                       148 |         2 |         180 |                1 | train      |
|         180 | short_graph_traversal   |                0.911111 |                    164 |                   0.905556 |                       163 |         5 |         180 |                1 | train      |
|         180 | simple_state_transition |                0.816667 |                    147 |                   0.777778 |                       140 |         3 |         180 |                1 | train      |
|         180 | variable_binding        |                0.894444 |                    161 |                   0.894444 |                       161 |         2 |         180 |                1 | train      |
|          80 | boolean_logic           |                0.925    |                     74 |                   0.85     |                        68 |         2 |          80 |                1 | validation |
|          80 | modular_arithmetic      |                0.725    |                     58 |                   0.725    |                        58 |         2 |          80 |                1 | validation |
|          80 | short_graph_traversal   |                0.9      |                     72 |                   0.8875   |                        71 |         5 |          80 |                1 | validation |
|          80 | simple_state_transition |                0.825    |                     66 |                   0.8      |                        64 |         3 |          80 |                1 | validation |
|          80 | variable_binding        |                0.8125   |                     65 |                   0.8125   |                        65 |         2 |          80 |                1 | validation |

Valid causal pairs: `1100`; final-test pairs: `250`.

No teacher-incorrect trajectory enters the paired causal or compression estimates.
