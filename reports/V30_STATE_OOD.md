# Unseen-State Causal Fidelity — V30

STATE-OOD uses unseen development/validation states with TOKEN_TRAIN pairs that were eligible in the source library. The future endpoint is the concatenated normalized signature from four probes selected before writes. Strong gate: cosine ≥0.9, magnitude [0.8,1.2], relative L2 ≤0.3, success fraction ≥0.5, ≥4/5 families.

### development

| condition | n | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| EXACT_REC_CONV | 60 | 0.976 | 0.978 | 0.220 |
| EXACT_CONV | 60 | 0.954 | 0.965 | 0.306 |
| EXACT_KV | 60 | 0.294 | 0.193 | 0.965 |
| GLOBAL_k32 | 60 | 0.864 | 0.912 | 0.509 |
| GLOBAL_k64 | 60 | 0.943 | 0.958 | 0.336 |
| FAMILY_k32 | 60 | 0.900 | 0.916 | 0.457 |
| FAMILY_k64 | 60 | 0.958 | 0.946 | 0.295 |
| LOFO_k32 | 60 | 0.844 | 0.882 | 0.547 |
| LOFO_k64 | 60 | 0.913 | 0.931 | 0.410 |

### validation

| condition | n | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| EXACT_REC_CONV | 120 | 0.976 | 0.963 | 0.220 |
| EXACT_CONV | 120 | 0.950 | 0.958 | 0.318 |
| EXACT_KV | 120 | 0.345 | 0.195 | 0.952 |
| GLOBAL_k32 | 120 | 0.878 | 0.912 | 0.487 |
| GLOBAL_k64 | 120 | 0.942 | 0.948 | 0.342 |
| FAMILY_k32 | 120 | 0.906 | 0.938 | 0.426 |
| FAMILY_k64 | 120 | 0.954 | 0.961 | 0.305 |
| LOFO_k32 | 120 | 0.856 | 0.879 | 0.527 |
| LOFO_k64 | 120 | 0.908 | 0.925 | 0.424 |

The fixed global k64 basis approaches but does not pass the L2 gate on state OOD; exact REC+Conv remains the native causal ceiling. Family-fixed and LOFO rows are separate controls, not replacements for one global basis.
