# Strict state replacement — V12

- Tested scope: `formal audit of absolute full persistent/model-cache replacement; writeback diagnostic retained separately`
- Raw target-state bypass detected: `False`
- Compact-only full-state verified: `False`
- Strict full-state replacement pass: `False`
- Reason: the frozen V12 representations encode persistent intervention deltas; they do not encode an absolute complete REC/conv/KV/model-cache state.

The diagnostic overwrites/reconstructs the intervention-bearing persistent channels through the strict prepared-state interface. It does **not** relabel this as complete model-cache replacement: unmodeled cache fields remain a structural scaffold. Consequently it cannot authorize H3 or an autonomous controller.

| method | h | direction | magnitude | semantic | output | sign | all-family gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| local_causal_basis_d384 | 1 | 0.915 | 0.963 | 0.634 | 0.885 | 0.820 | FAIL |
| local_causal_basis_d384 | 2 | 0.808 | 0.978 | 0.340 | 0.777 | 0.800 | FAIL |
| local_causal_basis_d384 | 4 | 0.672 | 1.004 | 0.202 | 0.643 | 0.756 | FAIL |
| local_causal_basis_d384 | 8 | 0.528 | 0.993 | 0.122 | 0.528 | 0.796 | FAIL |

Machine record: `results/v12/processed/strict_state_replacement_v12.parquet` (`1d0a1f1dcdd743cbbd190ab7015a8949e1c3534524f0d3c6b1bb195fe86b93f0`).
