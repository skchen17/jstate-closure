# Finite causal action primitive diagnostic — V16

A greedy dictionary of mean **train** single-action responses was selected without validation labels; held-out validation positive-single target-space linear projection coverage is:

|      K |   heldout_linear_response_coverage |
|-------:|-----------------------------------:|
|  2.000 |                              0.519 |
|  4.000 |                              0.669 |
|  8.000 |                              0.817 |
| 16.000 |                              0.901 |
| 32.000 |                              0.940 |

This is only a response-space coverage diagnostic. The candidate primitives were not independently tested as a complete semi-discrete controller, and mixtures/sequences were not validated across states; `FINITE_CAUSAL_ACTION_PRIMITIVE_REPRESENTATION_SUPPORTED` is **not** claimed. The selected indices at each K and exact coverage are in `v16_analysis.json`. No cognitive-primitive interpretation is made.
