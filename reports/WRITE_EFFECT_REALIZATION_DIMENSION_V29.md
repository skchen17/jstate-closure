# Write-Effect Realization Dimension — V29

A response-blind centered PCA basis was fitted to 90 natural REC+Conv token-write contrasts (25 calibration + 65 development), then causal donor-response reconstruction was tested on 10 held-out development and 50 validation pairs, both directions. This is an approximate **write-effect** test, not a compact persistent-state or dynamical-state test. The approximations are off-manifold and BF16-writeback audited.

| k | dev A←B / B←A rel-L2 | val A←B / B←A rel-L2 | strict gate |
|---:|---:|---:|---:|
| 4 | 0.226 / 0.271 | 0.217 / 0.203 | False |
| 8 | 0.215 / 0.241 | 0.213 / 0.182 | False |
| 16 | 0.196 / 0.215 | 0.168 / 0.163 | False |
| 32 | 0.181 / 0.157 | 0.146 / 0.133 | True |
| 64 | 0.165 / 0.147 | 0.123 / 0.116 | True |

Ambient REC+Conv write dimension `13369344`; train span rank `89`. Requested k `[128, 256]` exceed the available train span and were **not** called failures. No tested dimension is inferred beyond the strict gate.
