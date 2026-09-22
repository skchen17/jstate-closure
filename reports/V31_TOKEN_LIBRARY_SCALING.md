# Token-Library Scaling — V31

Frozen nested TRAIN-token prefixes 16/32/64/128/160 give pooled r95 39/65/109/184/220.

| tokens | observed writes | r95 |
|---|---|---|
| 128 | 808 | 184 |
| 16 | 112 | 39 |
| 160 | 1000 | 220 |
| 32 | 224 | 65 |
| 64 | 424 | 109 |

The chosen response-blind library has 160 TRAIN tokens plus 32 held-out AB tokens; the suggested 256-token point was not pre-registered as an executable library and has no causal result. No false 256-token extrapolation is made. Coverage changes both row count and token diversity; this is descriptive scaling.
