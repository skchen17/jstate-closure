# KV-Held-Fixed Recurrent Write — V29

In `REC+Conv` transfer, the eight attention layers' KV history remains **recipient-native**, including its new token slot. Yet future response moves reciprocally toward the donor in development and validation and the fixed independent final. This supports causal token-conditioned content in recurrent/conv persistent writes beyond direct KV token-history carryover. `KV` donor with recipient-native REC/Conv fails; the contrast is local to this design and endpoint.

| channel | role | A←B cosine / magnitude / rel-L2 | B←A cosine / magnitude / rel-L2 | reciprocal gate |
|---|---|---:|---:|---:|
| KV | development | 0.082 / 0.067 / 0.997 | 0.108 / 0.067 / 0.995 | False |
| KV | validation | 0.088 / 0.067 / 0.996 | 0.093 / 0.070 / 0.996 | False |
| REC+Conv | development | 0.998 / 0.995 / 0.067 | 0.998 / 0.997 / 0.067 | True |
| REC+Conv | validation | 0.998 / 0.996 / 0.070 | 0.998 / 0.996 / 0.067 | True |
