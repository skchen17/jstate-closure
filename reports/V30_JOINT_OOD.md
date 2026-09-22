# Joint State-and-Token OOD — V30

Primary difficult test: both target state and token contrast are outside the global basis fit. Each role uses 2 prospectively frozen TOKEN_VALIDATION contrasts per state.

### development

| condition | n | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| EXACT_REC_CONV | 60 | 0.961 | 0.965 | 0.280 |
| EXACT_CONV | 60 | 0.929 | 0.970 | 0.371 |
| EXACT_KV | 60 | 0.361 | 0.189 | 0.950 |
| GLOBAL_k32 | 60 | 0.791 | 0.848 | 0.619 |
| GLOBAL_k64 | 60 | 0.809 | 0.846 | 0.595 |
| FAMILY_k64 | 60 | 0.798 | 0.868 | 0.608 |
| LOFO_k64 | 60 | 0.789 | 0.824 | 0.618 |

### validation

| condition | n | cosine | magnitude | relative L2 |
|---|---|---|---|---|
| EXACT_REC_CONV | 120 | 0.976 | 0.969 | 0.220 |
| EXACT_CONV | 120 | 0.950 | 0.972 | 0.315 |
| EXACT_KV | 120 | 0.275 | 0.164 | 0.967 |
| GLOBAL_k32 | 120 | 0.809 | 0.817 | 0.594 |
| GLOBAL_k64 | 120 | 0.828 | 0.826 | 0.564 |
| FAMILY_k64 | 120 | 0.829 | 0.830 | 0.565 |
| LOFO_k64 | 120 | 0.814 | 0.826 | 0.591 |

Global k64 has validation median cosine 0.828, relative L2 0.564; exact REC+Conv is 0.220 L2. A fixed compact global basis is not causally sufficient here.
