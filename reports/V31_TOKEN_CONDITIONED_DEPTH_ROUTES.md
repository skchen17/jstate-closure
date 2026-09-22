# Token-Conditioned Depth Routes — V31

Frozen development-only route rule selected `FULL_CONV` as a **fallback**, not a passing route. All 24 Conv layers were retained.

| token category | development pairs | full-Conv L2 | best descriptive group | group L2 |
|---|---|---|---|---|
| CJK_SURFACE_UNRESOLVED | 1 | 0.345 | PREFIX_03 | 0.345 |
| LEXICAL_SURFACE | 6 | 0.393 | PREFIX_03 | 0.393 |
| PUNCTUATION_OR_STRUCTURE | 1 | 0.344 | PREFIX_03 | 0.344 |
| UNCLASSIFIED | 12 | 0.310 | PREFIX_03 | 0.310 |

Per-category minima are exploratory and were not promoted to validation after selection; sparse categories and failure of the full-Conv development gate preclude a token-conditioned depth-route claim. TRAIN energy differences do not substitute for predicted-group versus wrong/random matched-group causal superiority. V31-H is not established.
