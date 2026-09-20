# V19 natural versus finite-action raw-context utility

Same frozen V19 state split, same V18 train-frozen 128D J versus 128D J+REC+Conv+KV features, one hash-selected signed development action per state, same fixed ridge-1.0 136D design and V16 target-block normalization for both endpoints. Natural target is Y(P0,0,h); finite-action target is R(P0,a,h). Errors are divided by validation variation around the training target mean. This fixed linear comparison is not an exhaustive nonlinear ceiling.

| h | target | train | val | raw_gain natural | 95% CI | raw_gain action | 95% CI | Δgain |
|---|---|---|---|---|---|---|---|---|
| h1 | j | 400 | 100 | 0.0109 | [-0.0014, 0.0225] | 0.0064 | [-0.0173, 0.0292] | 0.0045 |
| h1 | stacked_normalized | 400 | 100 | -0.0018 | [-0.0145, 0.0098] | 0.0133 | [-0.0018, 0.0266] | -0.0151 |
| h2 | j | 50 | 15 | -0.0737 | [-0.1539, -0.0023] | -0.0432 | [-0.1088, 0.0175] | -0.0305 |
| h2 | stacked_normalized | 50 | 15 | -0.0721 | [-0.1502, -0.0117] | -0.0052 | [-0.0498, 0.0606] | -0.0669 |
| h4 | j | 50 | 15 | -0.0079 | [-0.0530, 0.0814] | -0.0481 | [-0.1041, 0.0355] | 0.0402 |
| h4 | stacked_normalized | 50 | 15 | -0.0303 | [-0.0967, 0.0830] | -0.0086 | [-0.1429, 0.1628] | -0.0218 |
| h8 | j | 50 | 15 | -0.0326 | [-0.0976, 0.0815] | -0.0018 | [-0.0436, 0.0724] | -0.0308 |
| h8 | stacked_normalized | 50 | 15 | -0.0325 | [-0.0874, 0.0809] | -0.0219 | [-0.1012, 0.0982] | -0.0106 |

h1 stack natural/action gain: -0.0018/0.0133. Raw predictive increment is distinct from causal N or M; no memory/state claim follows from gain alone.
