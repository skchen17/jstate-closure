# Correction Interception — V32

At the frozen layer-30 recurrent-operator output, replace Y11's component with its same-probe Y01 value and continue the suffix. The same input cache, token and position are retained; requested/realized output hashes are equal in all probes. Removed benefit is measured against the six-probe donor error.

| role | n | bidirectional successes | median removed | median restored | remove cosine | restore cosine | families | formal gate |
|---|---|---|---|---|---|---|---|---|
| development | 10 | 0 | 0.249 | 0.161 | 0.314 | 0.328 | 0 | False |
| validation | 10 | 0 | 0.188 | 0.083 | 0.284 | 0.297 | 0 | False |

No formal localized necessity claim is supported. A single candidate was selected prospectively; its failure does not establish that no other local site exists. Source: `site_intervention_*_v32.parquet`, `site_intervention_audit_*_v32.parquet`.
