# V40 — All Reports

Execution-complete formal V40 report bundle; calibration and formal outcomes are distinguished below.

<!-- V40_FROZEN_STARTING_POINT.md | SHA256 a73645e396de418ef8745335e9b6fc2b21e6848f8eeb4bcc526aaa7d5a658a8f -->

# V40 — Frozen Starting Point

V40 starts from the pushed V39 commit `0dea89c9296737ba713411c11de97184ba8c9eba` without changing V39 results or seals. V39's development interaction is background motivation, not a V40 outcome. Its primitive-mediator, validation, and independent-final branches remained unopened; V40 does not inherit their qualification.

The V40 base protocol is `artifacts/channel_function_v40.freeze.json`. During the first response-blind pool run, distinct state-transition programs were found to share recipient prompt text. No pool file and no model intervention outcome had been written. The base freeze was preserved. `artifacts/channel_function_v40_a1.freeze.json` is an append-only startup amendment that requires prompt-level collision rejection; all operative V40 pool and design seals use the `channel_function_v40_a1` prefix.

The amended sample-pool manifest at `data/v40/sample_pool_manifest_v40.json` records four disjoint roles: calibration 16, development 64, validation 32, and independent final 32 task states, each split evenly across the four user-specified families. Program and prompt hashes are disjoint across roles and from indexed historical sources. External answers were generated before model execution. Both model-specific tokenizer audits found exactly one differing natural token per paired prompt, with otherwise identical encoded prefix and suffix. No V40 model forward or intervention outcome was observed in constructing these records.

V40's primary model is Qwen3.5-4B; Falcon-H1-1.5B-Base is a comparison, not a required mechanism match. The experiment must not claim channel roles, decay lifetimes, or trajectory mediation until task-grounded causal runs and controls are recorded. The independent-final pool remains sealed against outcome inspection and selection.

---

<!-- V40_CALIBRATION_A3.md | SHA256 2634b35681ccf57838b10eac5edbf1bcaef53ed44b97eb0300349c805cb37d70 -->

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

---

<!-- V40_ANSWER_FORM_AUDIT.md | SHA256 6db39c524b568bda597f9145dfbec1e32c479507da5b04c102c281d5bc80c21b -->

# V40 — Answer-Form Calibration Audit

Calibration only. The two readouts compare answer logits immediately after
the prompt versus after one natural space token. All numbers below are
computed from saved calibration records; no development or held-out outcomes.

| Model | Family | Both native branches correct: direct | After one space |
|---|---|---:|---:|
| Q | boolean_logic | 2 | 2 |
| Q | long_context_dependency | 3 | 4 |
| Q | state_transition | 3 | 4 |
| Q | variable_binding | 4 | 4 |
| F | boolean_logic | 0 | 4 |
| F | long_context_dependency | 0 | 0 |
| F | state_transition | 0 | 0 |
| F | variable_binding | 0 | 0 |

| Model | Total direct | Total after one space |
|---|---:|---:|
| Q | 12 | 14 |
| F | 0 | 4 |

Answer formatting changes apparent native competence. Falcon's
Boolean family improves after a space, but its other families have no
both-correct calibration pairs under either readout. This is a scoring
diagnostic, not evidence for a channel role or a model deficit in general.
The formal answer readout and control design must be frozen before
development outcomes are observed.

---

<!-- V40_CHANNEL_FUNCTION_PROFILE.md | SHA256 620f52bd65044c7bd124cef288dfab3b07c76b4f9abe645b914cf0d76b54aa25 -->

# V40 — Channel Function Profile

The primary readout is the externally generated task answer after one space token.
Each row uses all sealed task states and separately reports the native-pair-qualified subset.
Calibration and V39 outcomes are not counted below.

| Model | Role | Complete / sealed | Native qualified | Condition | Donor recovered / all sealed | Donor recovered / qualified | Recovery Δ vs BASE, bootstrap 95% CI |
|---|---|---:|---:|---|---:|---:|---:|
| Q | development | 64/64 | 41 | BASE | 5/64 | 0/41 | 0.000 [0.000, 0.000] |
| Q | development | 64/64 | 41 | REC_only | 5/64 | 0/41 | 0.000 [0.000, 0.000] |
| Q | development | 64/64 | 41 | Conv_only | 19/64 | 11/41 | 0.268 [0.146, 0.415] |
| Q | development | 64/64 | 41 | KV_only | 34/64 | 27/41 | 0.659 [0.512, 0.780] |
| Q | development | 64/64 | 41 | REC_Conv | 21/64 | 12/41 | 0.293 [0.171, 0.439] |
| Q | development | 64/64 | 41 | REC_Conv_KV | 49/64 | 41/41 | 1.000 [1.000, 1.000] |
| Q | development | 64/64 | 41 | DONOR_NATIVE | 49/64 | 41/41 | 1.000 [1.000, 1.000] |
| F | development | 64/64 | 18 | BASE | 9/64 | 0/18 | 0.000 [0.000, 0.000] |
| F | development | 64/64 | 18 | REC_only | 9/64 | 0/18 | 0.000 [0.000, 0.000] |
| F | development | 64/64 | 18 | Conv_only | 13/64 | 4/18 | 0.222 [0.056, 0.444] |
| F | development | 64/64 | 18 | KV_only | 15/64 | 5/18 | 0.278 [0.110, 0.500] |
| F | development | 64/64 | 18 | REC_Conv | 15/64 | 5/18 | 0.278 [0.111, 0.500] |
| F | development | 64/64 | 18 | REC_Conv_KV | 29/64 | 18/18 | 1.000 [1.000, 1.000] |
| F | development | 64/64 | 18 | DONOR_NATIVE | 29/64 | 18/18 | 1.000 [1.000, 1.000] |
| Q | validation | 32/32 | 22 | BASE | 5/32 | 0/22 | 0.000 [0.000, 0.000] |
| Q | validation | 32/32 | 22 | REC_only | 5/32 | 0/22 | 0.000 [0.000, 0.000] |
| Q | validation | 32/32 | 22 | Conv_only | 11/32 | 5/22 | 0.227 [0.090, 0.409] |
| Q | validation | 32/32 | 22 | KV_only | 18/32 | 13/22 | 0.591 [0.364, 0.818] |
| Q | validation | 32/32 | 22 | REC_Conv | 13/32 | 7/22 | 0.318 [0.136, 0.500] |
| Q | validation | 32/32 | 22 | REC_Conv_KV | 27/32 | 22/22 | 1.000 [1.000, 1.000] |
| Q | validation | 32/32 | 22 | DONOR_NATIVE | 27/32 | 22/22 | 1.000 [1.000, 1.000] |
| F | validation | 32/32 | 10 | BASE | 1/32 | 0/10 | 0.000 [0.000, 0.000] |
| F | validation | 32/32 | 10 | REC_only | 1/32 | 0/10 | 0.000 [0.000, 0.000] |
| F | validation | 32/32 | 10 | Conv_only | 5/32 | 3/10 | 0.300 [0.100, 0.600] |
| F | validation | 32/32 | 10 | KV_only | 7/32 | 5/10 | 0.500 [0.200, 0.800] |
| F | validation | 32/32 | 10 | REC_Conv | 7/32 | 5/10 | 0.500 [0.200, 0.800] |
| F | validation | 32/32 | 10 | REC_Conv_KV | 12/32 | 10/10 | 1.000 [1.000, 1.000] |
| F | validation | 32/32 | 10 | DONOR_NATIVE | 12/32 | 10/10 | 1.000 [1.000, 1.000] |
| Q | independent_final | 32/32 | 19 | BASE | 3/32 | 0/19 | 0.000 [0.000, 0.000] |
| Q | independent_final | 32/32 | 19 | REC_only | 3/32 | 0/19 | 0.000 [0.000, 0.000] |
| Q | independent_final | 32/32 | 19 | Conv_only | 5/32 | 3/19 | 0.158 [0.000, 0.316] |
| Q | independent_final | 32/32 | 19 | KV_only | 17/32 | 15/19 | 0.789 [0.579, 0.947] |
| Q | independent_final | 32/32 | 19 | REC_Conv | 7/32 | 5/19 | 0.263 [0.105, 0.474] |
| Q | independent_final | 32/32 | 19 | REC_Conv_KV | 21/32 | 19/19 | 1.000 [1.000, 1.000] |
| Q | independent_final | 32/32 | 19 | DONOR_NATIVE | 21/32 | 19/19 | 1.000 [1.000, 1.000] |
| F | independent_final | 32/32 | 8 | BASE | 7/32 | 0/8 | 0.000 [0.000, 0.000] |
| F | independent_final | 32/32 | 8 | REC_only | 7/32 | 0/8 | 0.000 [0.000, 0.000] |
| F | independent_final | 32/32 | 8 | Conv_only | 8/32 | 1/8 | 0.125 [0.000, 0.375] |
| F | independent_final | 32/32 | 8 | KV_only | 12/32 | 4/8 | 0.500 [0.125, 0.875] |
| F | independent_final | 32/32 | 8 | REC_Conv | 11/32 | 4/8 | 0.500 [0.125, 0.875] |
| F | independent_final | 32/32 | 8 | REC_Conv_KV | 16/32 | 8/8 | 1.000 [1.000, 1.000] |
| F | independent_final | 32/32 | 8 | DONOR_NATIVE | 16/32 | 8/8 | 1.000 [1.000, 1.000] |

REC+Conv+KV reconstructs the complete native donor cache and serves as a positive control; its recovery is not independent evidence of channel specialization.
REC+Conv versus Conv-only is a conditional contrast, not an independent REC-storage proof.
The source-fork and shuffled-state controls are summarized in the task-grounded report.


## Secondary answer-alphabet logit-vector fidelity

This exploratory operationalization computes `1 − distance(condition, native donor) / distance(BASE, native donor)` after mean-centering the frozen answer-alphabet logits. Zero is BASE and one is the native donor; it is **not** full-vocabulary or hidden-state fidelity and does not establish semantic storage.

| Model | Role | Identifiable / native qualified | REC mean [95% CI] | Conv mean [95% CI] | KV mean [95% CI] | REC+Conv mean [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| Q | development | 41/41 | 0.003 [-0.002, 0.008] | 0.289 [0.237, 0.339] | 0.536 [0.480, 0.596] | 0.298 [0.249, 0.351] |
| F | development | 18/18 | 0.002 [-0.008, 0.012] | 0.286 [0.190, 0.410] | 0.527 [0.402, 0.648] | 0.294 [0.192, 0.414] |
| Q | validation | 22/22 | 0.008 [0.001, 0.014] | 0.342 [0.270, 0.410] | 0.526 [0.433, 0.625] | 0.351 [0.279, 0.420] |
| F | validation | 10/10 | 0.015 [0.004, 0.028] | 0.413 [0.247, 0.599] | 0.644 [0.488, 0.774] | 0.437 [0.274, 0.611] |
| Q | independent_final | 19/19 | 0.011 [0.003, 0.019] | 0.300 [0.238, 0.361] | 0.622 [0.522, 0.719] | 0.314 [0.250, 0.378] |
| F | independent_final | 8/8 | 0.008 [-0.003, 0.019] | 0.365 [0.253, 0.532] | 0.535 [0.283, 0.755] | 0.404 [0.289, 0.553] |

---

<!-- V40_TASK_GROUNDED_RESULTS.md | SHA256 072a5116202e2c1c82c0b786a8b3745554866c08be0f8f6c55cff86176dbe446 -->

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

---

<!-- V40_TEMPORAL_PERSISTENCE.md | SHA256 12e78bad7b0c93d9478cf9757f5cc3fcb049ae61009de8ccda66f3982c78f390 -->

# V40 — Temporal Persistence

Qwen was primary. Each delay inserts fixed shared neutral tokens before the common task suffix.
Effects are empirical; no exponential decay law is assumed.

| Role | Delay | Complete / sealed | Native qualified | REC recovery (all; qualified) | Conv recovery (all; qualified) | KV recovery (all; qualified) | REC+Conv recovery (all; qualified) |
|---|---:|---:|---:|---:|---:|---:|---:|
| development | 0 | 64/64 | 41 | 5/64; 0/41 | 19/64; 11/41 | 34/64; 27/41 | 21/64; 12/41 |
| development | 1 | 64/64 | 44 | 4/64; 0/44 | 17/64; 13/44 | 35/64; 29/44 | 19/64; 15/44 |
| development | 2 | 64/64 | 44 | 4/64; 0/44 | 7/64; 3/44 | 41/64; 37/44 | 10/64; 6/44 |
| development | 4 | 64/64 | 44 | 4/64; 0/44 | 8/64; 2/44 | 43/64; 37/44 | 10/64; 4/44 |
| development | 8 | 64/64 | 42 | 4/64; 0/42 | 6/64; 2/42 | 43/64; 38/42 | 7/64; 2/42 |
| development | 16 | 64/64 | 38 | 4/64; 0/38 | 5/64; 2/38 | 35/64; 31/38 | 7/64; 3/38 |
| development | 32 | 64/64 | 27 | 5/64; 0/27 | 4/64; 1/27 | 29/64; 24/27 | 6/64; 2/27 |
| validation | 0 | 32/32 | 22 | 5/32; 0/22 | 11/32; 5/22 | 18/32; 13/22 | 13/32; 7/22 |
| validation | 1 | 32/32 | 22 | 5/32; 0/22 | 8/32; 3/22 | 16/32; 13/22 | 13/32; 8/22 |
| validation | 2 | 32/32 | 21 | 5/32; 0/21 | 7/32; 2/21 | 21/32; 17/21 | 8/32; 3/21 |
| validation | 4 | 32/32 | 21 | 5/32; 0/21 | 6/32; 1/21 | 23/32; 19/21 | 6/32; 1/21 |
| validation | 8 | 32/32 | 20 | 5/32; 0/20 | 5/32; 0/20 | 21/32; 17/20 | 7/32; 2/20 |
| validation | 16 | 32/32 | 19 | 6/32; 1/19 | 5/32; 1/19 | 19/32; 16/19 | 5/32; 1/19 |
| validation | 32 | 32/32 | 12 | 5/32; 0/12 | 4/32; 0/12 | 15/32; 11/12 | 4/32; 0/12 |
| independent_final | 0 | 32/32 | 19 | 3/32; 0/19 | 5/32; 3/19 | 17/32; 15/19 | 7/32; 5/19 |
| independent_final | 1 | 32/32 | 22 | 3/32; 0/22 | 9/32; 6/22 | 15/32; 13/22 | 9/32; 6/22 |
| independent_final | 2 | 32/32 | 22 | 1/32; 0/22 | 5/32; 4/22 | 20/32; 19/22 | 5/32; 4/22 |
| independent_final | 4 | 32/32 | 23 | 1/32; 0/23 | 2/32; 1/23 | 19/32; 18/23 | 3/32; 2/23 |
| independent_final | 8 | 32/32 | 22 | 1/32; 0/22 | 3/32; 2/22 | 20/32; 19/22 | 3/32; 2/22 |
| independent_final | 16 | 32/32 | 22 | 2/32; 0/22 | 2/32; 1/22 | 19/32; 18/22 | 2/32; 1/22 |
| independent_final | 32 | 32/32 | 14 | 3/32; 0/14 | 2/32; 0/14 | 14/32; 12/14 | 2/32; 0/14 |

Independent-final median answer-margin effect as a fraction of the contemporaneous native-donor ceiling:

| Delay | REC | Conv | KV | REC+Conv |
|---:|---:|---:|---:|---:|
| 0 | 0.011 | 0.382 | 0.711 | 0.386 |
| 1 | 0.040 | 0.319 | 0.607 | 0.352 |
| 2 | 0.022 | 0.188 | 0.783 | 0.224 |
| 4 | 0.028 | 0.167 | 0.828 | 0.186 |
| 8 | 0.032 | 0.123 | 0.787 | 0.177 |
| 16 | 0.022 | 0.124 | 0.760 | 0.161 |
| 32 | 0.000 | 0.062 | 0.914 | 0.074 |

Observed-grid half-crossing uses only states native-qualified at every delay and requires at least 16 such states.

| Role | Common qualified | Channel | First persistent half-crossing |
|---|---:|---|---:|
| development | 24 | Conv_only | 4 |
| development | 24 | KV_only | not identifiable |
| development | 24 | REC_Conv | 4 |
| development | 24 | REC_Conv_KV | not identifiable |
| development | 24 | REC_only | not identifiable |
| validation | 12 | Conv_only | not identifiable |
| validation | 12 | KV_only | not identifiable |
| validation | 12 | REC_Conv | not identifiable |
| validation | 12 | REC_Conv_KV | not identifiable |
| validation | 12 | REC_only | not identifiable |
| independent_final | 12 | Conv_only | not identifiable |
| independent_final | 12 | KV_only | not identifiable |
| independent_final | 12 | REC_Conv | not identifiable |
| independent_final | 12 | REC_Conv_KV | not identifiable |
| independent_final | 12 | REC_only | not identifiable |

A grid crossing is not a continuous or universal channel half-life.


## Independent-final task-family profiles (descriptive appendix)

Each cell uses the native-qualified states of that family at that delay; denominators may change with delay. These subgroup comparisons are exploratory.

| Family | Delay | Qualified / sealed | REC | Conv | KV | REC+Conv |
|---|---:|---:|---:|---:|---:|---:|
| boolean_logic | 0 | 4/8 | 0/4 | 0/4 | 4/4 | 0/4 |
| boolean_logic | 1 | 4/8 | 0/4 | 0/4 | 4/4 | 0/4 |
| boolean_logic | 2 | 4/8 | 0/4 | 2/4 | 4/4 | 0/4 |
| boolean_logic | 4 | 4/8 | 0/4 | 0/4 | 4/4 | 0/4 |
| boolean_logic | 8 | 4/8 | 0/4 | 1/4 | 4/4 | 1/4 |
| boolean_logic | 16 | 4/8 | 0/4 | 0/4 | 4/4 | 0/4 |
| boolean_logic | 32 | 4/8 | 0/4 | 0/4 | 4/4 | 0/4 |
| long_context_dependency | 0 | 5/8 | 0/5 | 3/5 | 2/5 | 5/5 |
| long_context_dependency | 1 | 8/8 | 0/8 | 6/8 | 0/8 | 6/8 |
| long_context_dependency | 2 | 8/8 | 0/8 | 2/8 | 5/8 | 4/8 |
| long_context_dependency | 4 | 8/8 | 0/8 | 1/8 | 6/8 | 2/8 |
| long_context_dependency | 8 | 8/8 | 0/8 | 1/8 | 7/8 | 1/8 |
| long_context_dependency | 16 | 8/8 | 0/8 | 1/8 | 6/8 | 1/8 |
| long_context_dependency | 32 | 2/8 | 0/2 | 0/2 | 2/2 | 0/2 |
| state_transition | 0 | 2/8 | 0/2 | 0/2 | 1/2 | 0/2 |
| state_transition | 1 | 2/8 | 0/2 | 0/2 | 1/2 | 0/2 |
| state_transition | 2 | 2/8 | 0/2 | 0/2 | 2/2 | 0/2 |
| state_transition | 4 | 4/8 | 0/4 | 0/4 | 2/4 | 0/4 |
| state_transition | 8 | 3/8 | 0/3 | 0/3 | 2/3 | 0/3 |
| state_transition | 16 | 3/8 | 0/3 | 0/3 | 2/3 | 0/3 |
| state_transition | 32 | 2/8 | 0/2 | 0/2 | 1/2 | 0/2 |
| variable_binding | 0 | 8/8 | 0/8 | 0/8 | 8/8 | 0/8 |
| variable_binding | 1 | 8/8 | 0/8 | 0/8 | 8/8 | 0/8 |
| variable_binding | 2 | 8/8 | 0/8 | 0/8 | 8/8 | 0/8 |
| variable_binding | 4 | 7/8 | 0/7 | 0/7 | 6/7 | 0/7 |
| variable_binding | 8 | 7/8 | 0/7 | 0/7 | 6/7 | 0/7 |
| variable_binding | 16 | 7/8 | 0/7 | 0/7 | 6/7 | 0/7 |
| variable_binding | 32 | 6/8 | 0/6 | 0/6 | 5/6 | 0/6 |

Bootstrap margin estimates for every family, delay, and channel are in the sealed machine appendix.

---

<!-- V40_CHANNEL_COMPARISON.md | SHA256 d973bae71d1bce56db38fc393784dbf9037708bc3de2c9bf1b9cfdd35c3493a5 -->

# V40 — Channel Comparison

Qwen is the primary model; Falcon is a descriptive comparison when its paired native-answer baseline is insufficient.
Identical channel roles were never required.

| Role | Qwen qualified | Qwen gate | Falcon qualified | Falcon scope |
|---|---:|---|---:|---|
| development | 41/64 | True | 18/64 | descriptive only; no preregistered Falcon gate |
| validation | 22/32 | True | 10/32 | descriptive only; no preregistered Falcon gate |
| independent_final | 19/32 | True | 8/32 | descriptive only; no preregistered Falcon gate |

A weak Falcon baseline here is an assay limitation, not a general claim about Falcon ability or its channel ontology.

---

<!-- V40_TRAJECTORY_CAUSALITY.md | SHA256 3dbbc848b7aaa57524a44f7177709ba7b688078fdde2a0512dbe71c46242b6e4 -->

# V40 — Trajectory Causality

Qwen early/middle/late REC groups are fixed thirds of its 24 recurrent layers.
Each factorial condition starts from recipient REC/KV with donor Conv as a common background; all future computation runs naturally.
The primary three-way contrast is on the externally labelled donor-versus-recipient answer margin.

| Role | Delay | Complete / sealed | Qualified | R000 donor recovery (all; qualified) | R111 donor recovery (all; qualified) | Three-way mean [95% CI] | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| development | 0 | 64/64 | 41 | 19/64; 11/41 | 21/64; 12/41 | 0.009 [-0.037, 0.050] | False |
| development | 8 | 64/64 | 42 | 6/64; 2/42 | 7/64; 2/42 | 0.007 [-0.030, 0.046] | False |
| validation | 0 | 32/32 | 22 | 11/32; 5/22 | 13/32; 7/22 | 0.010 [-0.052, 0.068] | False |
| validation | 8 | 32/32 | 20 | 5/32; 0/20 | 7/32; 2/20 | -0.056 [-0.113, 0.006] | False |
| independent_final | 0 | 32/32 | 19 | 5/32; 3/19 | 7/32; 5/19 | -0.018 [-0.056, 0.021] | False |
| independent_final | 8 | 32/32 | 22 | 3/32; 2/22 | 3/32; 2/22 | -0.017 [-0.057, 0.020] | False |

## Independent-final task-family interaction

| Delay | Family | Qualified states | Median three-way answer-margin interaction |
|---:|---|---:|---:|
| 0 | boolean_logic | 4 | 0.000 |
| 0 | long_context_dependency | 5 | 0.000 |
| 0 | state_transition | 2 | 0.016 |
| 0 | variable_binding | 8 | 0.000 |
| 8 | boolean_logic | 4 | -0.062 |
| 8 | long_context_dependency | 8 | 0.000 |
| 8 | state_transition | 3 | 0.031 |
| 8 | variable_binding | 7 | 0.000 |

## Final-answer hidden geometry at the last recurrent checkpoint

| Delay | Condition | Median cosine to native donor |
|---:|---|---:|
| 0 | R000 | 0.9515 |
| 0 | R100 | 0.9522 |
| 0 | R010 | 0.9546 |
| 0 | R001 | 0.9516 |
| 0 | R111 | 0.9546 |
| 8 | R000 | 0.9626 |
| 8 | R100 | 0.9625 |
| 8 | R010 | 0.9670 |
| 8 | R001 | 0.9623 |
| 8 | R111 | 0.9671 |

Hidden-state cosine summaries and checkpoint REC/Conv norms are in the sealed machine records.
Donor-like hidden geometry without task recovery is not task-grounded mediation.
This factorial tests depth composition on one channel background; it does not identify a unique primitive or all token-time mechanisms.

---

<!-- V40_COMPLETE_REPORT.md | SHA256 838aa7bd4b59a66e53eaf4d852c2ef1a6a7ec69d6994d17b9f98793463466282 -->

# V40 — Complete Report

Functional and Temporal Decomposition of Persistent State in Hybrid Language Models

V40 is execution-complete across its sealed Qwen functional, temporal, and trajectory branches and Falcon functional comparison.
V39 and the original V40 calibration are unchanged historical inputs, not formal V40 outcomes.

## Frozen design and inference scope

The state is the independent unit. Task labels were generated before model execution; four state-ID-disjoint pools contain calibration=16, development=64, validation=32, independent_final=32 states.
Qwen is primary and Falcon is descriptive comparison. The fixed readout is after_one_space; uncertainty uses 2000 state-unit bootstrap resamples and 95% percentile intervals.
Formal function, delayed continuation, and REC-depth factorial outcomes are reported separately; controls and invalid/unavailable rows remain visible. The answer-alphabet vector metric is secondary and exploratory, not a substitute for externally labelled recovery.

## Scientific assessment

- Qwen independent-final task-grounded REC+Conv donor-recovery advantage over both matched mismatch/shuffled controls, positive 95% lower bounds: **True**.
- Qwen independent-final task-grounded positive REC three-way depth interaction at both tested delays: **False**.
- Channel-specific empirical delay profiles and any identifiable observed-grid half-crossings are in `V40_TEMPORAL_PERSISTENCE.md`; no universal decay law or semantic storage role is inferred.
- Falcon remains a comparison only where its native paired-answer baseline is interpretable; no shared or model-specific ontology is assumed from an inadequate assay.

The broad V40 hypothesis succeeds only if channel roles, temporal specialization, trajectory causality, and external task grounding are each supported. A partial or negative result is a completed experiment, not proof of the untested claims.
See the five topic reports and `V40_ALL_REPORTS.md` for the full version-local account.

## Answer to the V40 scientific question

**Functional content under this assay.** Qwen completed 32/32 states; 19 were native-pair-qualified and the frozen Qwen gate was True. Qualified donor-answer recovery: REC_only=0/19, Conv_only=3/19, KV_only=15/19, REC_Conv=5/19, REC_Conv_KV=19/19. This is a causal channel profile on generated tasks, not a proof that a tensor stores a semantic variable.

**Task grounding.** REC+Conv versus matched same-prefix mismatch: 0.267 [0.067, 0.468]; versus shuffled natural REC+Conv: 0.263 [0.105, 0.474]. The incremental REC+Conv-versus-Conv recovery contrast is 0.105 [0.000, 0.263]; a CI including zero does not establish a separate REC answer contribution.

**Temporal persistence.** Qwen native-qualified recovery by delay: REC_only 0/19 at delay 0 and 0/14 at delay 32, Conv_only 3/19 at delay 0 and 0/14 at delay 32, KV_only 15/19 at delay 0 and 12/14 at delay 32, REC_Conv 5/19 at delay 0 and 0/14 at delay 32. These per-delay denominators change, so the rates alone are not a same-cohort decay curve. Common native-qualified states across all delays: 12; observed-grid half-crossings: Conv_only=not identifiable, KV_only=not identifiable, REC_Conv=not identifiable, REC_Conv_KV=not identifiable, REC_only=not identifiable. No continuous or universal half-life is inferred.

**Depth trajectory.** Three-way external-answer margin interaction at delay 0: -0.018 [-0.056, 0.021] (n=19); at delay 8: -0.017 [-0.057, 0.020] (n=22). The predeclared gates were False and False. Donor-like hidden geometry without answer recovery is not mediation.

**Cross-model limit.** Falcon completed 32/32 states but only 8 were native-pair-qualified under this readout; its channel counts are descriptive and cannot establish a common architecture-independent role.

**Overall.** V40 execution is complete, but its full four-part scientific success criterion is not established because the task-grounded depth-trajectory gate failed. Any positive functional and empirical temporal profiles remain bounded to these generated tasks, model versions, readout, delays, and tested state pool.

---
