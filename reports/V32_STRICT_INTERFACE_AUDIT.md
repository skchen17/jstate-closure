# Strict Interface Audit — V32

| role | factorial rows | native exact | site probe patches | site exact | diagnostic probe patches | diagnostic exact |
|---|---|---|---|---|---|---|
| development | 125 | True | 60 | True | 300 | True |
| validation | 60 | True | 60 | True | 300 | True |

Incoming prefix and cache-channel hashes, donor/recipient full-state hashes, token-pair and six-probe hashes, target Conv and recipient KV hashes, layer-group hashes, requested/realized component hashes, cache lengths and role IDs are in the design and raw audit records. Hooks are context-managed and removed even on failure. V32 final-state response count is zero. The factorial audit's non-outcome `field_count` metadata incorrectly records 120 for the three conditions; an append-only `audit_metadata_amendment_v32.json` corrects this to 24+24+48=96 without altering the raw records, exact equality checks or outcomes. The trace lacks a direct convolution-output and recurrent-kernel update interception; those are explicitly untested, not silently described as measured.
