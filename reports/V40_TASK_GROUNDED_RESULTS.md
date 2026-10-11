# V40 — Task-Grounded Results

Labels were produced by the sealed task generator before model execution. The state is the independent unit.
The frozen endpoint is constrained one-token answer selection over each family's finite alphabet, not unrestricted text generation.
An intervention is not called semantically successful from cosine similarity alone.

| Model | Role | Sealed | Complete | Native qualified | BASE donor recovery | REC+Conv donor recovery | Native donor recovery |
|---|---|---:|---:|---:|---:|---:|---:|
| Q | development | 64 | 64 | 41 | 0/41 | 12/41 | 41/41 |
| F | development | 64 | 64 | 18 | 0/18 | 5/18 | 18/18 |
| Q | validation | 32 | 32 | 22 | 0/22 | 7/22 | 22/22 |
| F | validation | 32 | 32 | 10 | 0/10 | 5/10 | 10/10 |
| Q | independent_final | 32 | 32 | 19 | 0/19 | 5/19 | 19/19 |
| F | independent_final | 32 | 32 | 8 | 0/8 | 4/8 | 8/8 |

## Qwen development by task family

| Family | Sealed | Qualified | REC | Conv | KV | REC+Conv |
|---|---:|---:|---:|---:|---:|---:|
| boolean_logic | 16 | 9 | 0 | 0 | 9 | 0 |
| long_context_dependency | 16 | 12 | 0 | 10 | 2 | 11 |
| state_transition | 16 | 4 | 0 | 0 | 1 | 0 |
| variable_binding | 16 | 16 | 0 | 1 | 15 | 1 |

## Qwen validation by task family

| Family | Sealed | Qualified | REC | Conv | KV | REC+Conv |
|---|---:|---:|---:|---:|---:|---:|
| boolean_logic | 8 | 5 | 0 | 0 | 5 | 0 |
| long_context_dependency | 8 | 6 | 0 | 3 | 1 | 5 |
| state_transition | 8 | 3 | 0 | 0 | 0 | 0 |
| variable_binding | 8 | 8 | 0 | 2 | 7 | 2 |

## Qwen independent_final by task family

| Family | Sealed | Qualified | REC | Conv | KV | REC+Conv |
|---|---:|---:|---:|---:|---:|---:|
| boolean_logic | 8 | 4 | 0 | 0 | 4 | 0 |
| long_context_dependency | 8 | 5 | 0 | 3 | 2 | 5 |
| state_transition | 8 | 2 | 0 | 0 | 1 | 0 |
| variable_binding | 8 | 8 | 0 | 0 | 8 | 0 |

## Q controls (independent_final)

| Control | Valid rows | Unavailable | Task-correct rows | Paired-donor rows |
|---|---:|---:|---:|---:|
| irrelevant_history | 96 | 0 | 81 | 81 |
| same_prefix_mismatch | 48 | 8 | 19 | 3 |
| shuffled_REC_Conv | 32 | 0 | 22 | 3 |
| surface_only | 96 | 0 | 78 | 78 |

Paired REC+Conv donor-recovery advantage over matched negative controls (state bootstrap):
- same_prefix_mismatch: 0.267 [0.067, 0.468] (n=15).
- shuffled_REC_Conv: 0.263 [0.105, 0.474] (n=19).

Split/full constrained-answer agreement: 63/64; maximum full-vocabulary logit difference: 0.4062.

## F controls (independent_final)

| Control | Valid rows | Unavailable | Task-correct rows | Paired-donor rows |
|---|---:|---:|---:|---:|
| irrelevant_history | 96 | 0 | 12 | 12 |
| same_prefix_mismatch | 48 | 8 | 5 | 14 |
| shuffled_REC_Conv | 32 | 0 | 13 | 7 |
| surface_only | 96 | 0 | 32 | 32 |

Paired REC+Conv donor-recovery advantage over matched negative controls (state bootstrap):
- same_prefix_mismatch: — (n=0).
- shuffled_REC_Conv: 0.500 [0.125, 0.875] (n=8).

Split/full constrained-answer agreement: 62/64; maximum full-vocabulary logit difference: 0.8750.

For surface/history controls the donor and recipient external answers are identical, so their paired-donor column is merely unchanged-label correctness, not a transfer effect.
Controls are diagnostics on frozen natural states; absent baseline competence bounds semantic interpretation.


## Native split/full prediction sensitivity

The primary analysis is unchanged; the strict subset excludes any state with a native BASE or DONOR_NATIVE constrained-answer disagreement.

| Model | Role | Split-discordant states | Original qualified | Strict qualified | Strict Qwen gate | Strict REC+Conv recovery |
|---|---|---:|---:|---:|---|---:|
| Q | development | 2 | 41 | 40 | True | 11/40 |
| F | development | 2 | 18 | 18 | None | 5/18 |
| Q | validation | 0 | 22 | 22 | True | 7/22 |
| F | validation | 0 | 10 | 10 | None | 5/10 |
| Q | independent_final | 1 | 19 | 19 | None | 5/19 |
| F | independent_final | 2 | 8 | 8 | None | 4/8 |
