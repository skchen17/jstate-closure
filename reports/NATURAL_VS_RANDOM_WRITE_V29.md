# Natural Versus Random Write — V29

Ten states per role compare exact native REC+Conv donor writes with random same-norm, within-field shuffled, sign-flipped, and wrong-incoming-state delta controls. Artificial controls are off-manifold and BF16-realization ratios are saved per row; they are not substitutes for native donor-field transfer. Natural writes are much more donor-faithful than the random/shuffled/sign-flipped controls. The wrong-state control can retain partial directionality, so it is not evidence of a uniquely state-specific code.

| role | condition | median donor cosine | median relative L2 |
|---|---|---:|---:|
| development | natural_REC+Conv | 0.998 | 0.064 |
| development | random_same_norm | 0.341 | 1.389 |
| development | shuffled_natural | 0.426 | 1.022 |
| development | sign_flipped | -0.172 | 1.182 |
| development | wrong_state | 0.875 | 0.507 |
| validation | natural_REC+Conv | 0.998 | 0.071 |
| validation | random_same_norm | 0.417 | 1.430 |
| validation | shuffled_natural | 0.415 | 1.223 |
| validation | sign_flipped | -0.186 | 1.116 |
| validation | wrong_state | 0.972 | 0.254 |
