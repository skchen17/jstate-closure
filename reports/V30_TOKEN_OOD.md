# Unseen-Token Causal Fidelity — V30

TOKEN-OOD tests 10 seen training states × 2 TOKEN_VALIDATION pairs each; no held-out token pair entered basis fitting.

| condition | n | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| EXACT_REC_CONV | 20 | 0.977 | 0.977 | 0.214 |
| GLOBAL_k32 | 20 | 0.777 | 0.840 | 0.630 |
| GLOBAL_k64 | 20 | 0.819 | 0.846 | 0.589 |
| FAMILY_k64 | 20 | 0.836 | 0.850 | 0.564 |
| LOFO_k64 | 20 | 0.788 | 0.829 | 0.621 |

Even on seen states, k64 fails the causal gate on new token contrasts. Independent validation of unseen state plus unseen token is reported under JOINT-OOD. TOKEN_FINAL remains unopened.
