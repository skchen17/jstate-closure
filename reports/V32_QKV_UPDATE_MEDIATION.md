# QKV and Update Mediation — V32

The architecture exposes a joint pre-convolution qkv projection and a normalized recurrent read. The q/k/v post-convolution split and fused delta-update were not individually writable with the audited interface. Exact diagnostic patches at the frozen primary layer yield:

| role | component | median removed | median restored | remove cosine | restore cosine |
|---|---|---|---|---|---|
| development | normalized_recurrent_read | 0.249 | 0.161 | 0.314 | 0.328 |
| development | qkv_projection | 0.026 | -0.010 | -0.019 | -0.014 |
| validation | normalized_recurrent_read | 0.188 | 0.083 | 0.284 | 0.297 |
| validation | qkv_projection | 0.020 | -0.024 | -0.021 | -0.033 |

These checks do not isolate q, k, v, beta or the update term, and the component list was not part of the original primary candidate-selection rule. V32-F is unconfirmed, not falsified for untested update variables.
