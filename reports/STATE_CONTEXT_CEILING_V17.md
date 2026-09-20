# V17 state-context ceiling

Protocol `71c0501d3289b4ecafa5957c3121a076aee665e8967db7890ff19c3c050b77f8`; split `5096bead69209ec4a46747866ead84e5b99942f06f179216e0efd0e87d4682b9`. Reused V16 reliable single-action responses on 100 train and 50 untouched development-validation states; 2796 validation action rows and 58 eligible coordinate/sign keys. All context inputs are *clean current* V13 REC/Conv/KV or clean current J; neither perturbed cache nor future label entered a predictor. Raw channels use full measured linear kernels, train-only centering/scaling, and identical per-action kernel-ridge capacity. Lambda `0.1` was chosen on training states only. This is a linear-kernel reference, **not** an unlimited nonlinear oracle.

| predictor | J rel L2 | stack rel L2 | J direction median | stack direction median |
|---|---:|---:|---:|---:|
| action_only | 0.5314 | 0.5764 | 0.8633 | 0.8276 |
| j | 0.2989 | 0.3518 | 0.9601 | 0.9505 |
| j+rec | 0.2828 | 0.3403 | 0.9638 | 0.9548 |
| j+conv | 0.2865 | 0.3424 | 0.9634 | 0.9543 |
| j+kv | 0.2949 | 0.3491 | 0.9602 | 0.9515 |
| j+rec+conv | 0.2838 | 0.3408 | 0.9638 | 0.9548 |
| j+rec+kv | 0.2879 | 0.3425 | 0.9622 | 0.9533 |
| j+conv+kv | 0.2895 | 0.3429 | 0.9616 | 0.9530 |
| j+rec+conv+kv | 0.2863 | 0.3406 | 0.9627 | 0.9539 |
| full_raw_reference | 0.2800 | 0.3377 | 0.9642 | 0.9566 |

The predeclared materiality gate required **both** J and stack absolute rel-L2 gain ≥ `0.05` plus a positive paired state-bootstrap lower bound. J+all improved J by `0.0126` (95% CI `[0.00791060309077974, 0.01681136587834107]`) and stack by `0.0113` (95% CI `[0.002569702453078568, 0.021050114743440172]`). The raw-only reference improved J by `0.0189` and stack by `0.0142`; neither reaches 0.05. **Material ceiling gate failed.** Small positive differences do not justify V17-A (J truly sufficient) or V17-B (material raw dependence).

Family-wise stack error is heterogeneous; one family worsens when all channels are added:

| family | J-only stack rel L2 | J+all stack rel L2 | gain |
|---|---:|---:|---:|
| boolean_logic | 0.2597 | 0.2451 | 0.0146 |
| modular_arithmetic | 0.4286 | 0.4186 | 0.0100 |
| short_graph_traversal | 0.3610 | 0.3682 | -0.0072 |
| simple_state_transition | 0.4036 | 0.3781 | 0.0255 |
| variable_binding | 0.3114 | 0.2897 | 0.0217 |

The kernel is limited to 99 nonzero centered train-state directions for every source. Equal kernel-ridge state count and lambda control estimator capacity, but this comparison does not rule out a different nonlinear state×action model discovering larger gain. Response predictions are calibrated per coordinate/sign and actual finite-action alpha; only the V16 reliable single-action subset is covered here, not pair/dense or a new independent bank.
