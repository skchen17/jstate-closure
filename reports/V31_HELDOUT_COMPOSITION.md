# Held-Out Composition — V31

The primary test held out the AB *single token* while its A and B constituents were individually in TRAIN. All three forks start from the same prefix cache. The 20 unseen development states × two frozen AB combinations yield 40 tests; no a→b+b→c tensor identity was used.

| condition | n | cosine | magnitude | causal L2 | success fraction | pass |
|---|---|---|---|---|---|---|
| EXACT_REC_CONV | 40 | 0.967 | 0.975 | 0.255 | 0.65 | False |
| UNIT_ADDITIVE | 40 | 0.767 | 1.093 | 0.731 | 0.0 | False |
| GLOBAL_SCALAR_GATED | 40 | 0.751 | 0.776 | 0.662 | 0.0 | False |
| LOW_ORDER_INTERACTION | 40 | 0.752 | 0.777 | 0.662 | 0.0 | False |
| STATE_CONDITIONED_SCALAR_GATED | 40 | 0.746 | 0.760 | 0.667 | 0.0 | False |
| A_ONLY | 40 | 0.535 | 0.789 | 0.908 | 0.0 | False |
| B_ONLY | 40 | 0.772 | 0.889 | 0.650 | 0.0 | False |
| SIGN_FLIPPED_B | 40 | -0.015 | 0.475 | 1.123 | 0.0 | False |

No additive or low-order composition passes. Even the exact REC+Conv donor-field transplant misses the strict four-of-five-family development gate (3/5); this ceiling is not relabeled a compositional model. Frozen selection disallows trying new formulas on validation or independent final. This is failure of the *tested surface-composition proxy*, not proof that meaningful semantic primitives do not exist.
