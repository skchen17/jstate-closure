# V40 — Sequential Native-Cache Calibration

Status: calibration only; no development, validation, or independent-final outcomes.

The a2 batched-cache pilots remain historical diagnostics. The a3 runs use native
single-token continuation and exact native channel cache swaps. All counts and
numbers below are generated from the saved a3 JSONL records.

| Model | Family | States | Recipient native correct | Donor native correct | Both correct | REC donor recovery | Conv donor recovery | KV donor recovery | REC+Conv donor recovery | All-channel donor recovery |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q | boolean_logic | 4 | 3 | 3 | 2 | 1 | 1 | 3 | 1 | 3 |
| Q | long_context_dependency | 4 | 3 | 4 | 3 | 1 | 3 | 1 | 3 | 4 |
| Q | state_transition | 4 | 3 | 4 | 3 | 1 | 2 | 1 | 3 | 4 |
| Q | variable_binding | 4 | 4 | 4 | 4 | 0 | 3 | 2 | 3 | 4 |
| F | boolean_logic | 4 | 1 | 2 | 0 | 3 | 3 | 3 | 2 | 2 |
| F | long_context_dependency | 4 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| F | state_transition | 4 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| F | variable_binding | 4 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |

Native-write audit: all intervention rows have exact requested and untouched-field proofs.

| Model | Exact intervention rows | Max split/full logit difference | Split/full constrained-prediction agreement |
|---|---:|---:|---:|
| Q | 80 | 0.4688 | 1.000 |
| F | 80 | 1.0000 | 1.000 |

These calibration numbers are diagnostic. A channel role, lifetime, or
trajectory mechanism requires a separately frozen formal design and held-out
task-grounded outcomes. Poor native answer recovery limits interpretation.
