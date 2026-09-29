# V40 — All Reports

Calibration checkpoint only; no formal development or held-out intervention outcomes.

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

<!-- V40_CHANNEL_FUNCTION_PROFILE.md | SHA256 31bdedc170d76554102bb6ebc2216cdbfc9915c7de3da5db5cb6191077a90bce -->

# V40 — Channel Function Profile

Status: **calibration-only interventions recorded; formal development not run**.

For each externally labelled natural token fork, the frozen condition map is BASE, REC-only, Conv-only, KV-only, REC+Conv, and REC+Conv+KV. The recipient/donor pair differs in one encoded token, and labels are fixed by the task generator. The proposed primary comparison is recovery of the donor versus recipient task variable after native cache-channel exchange, not donor-response cosine. Donor-vector fidelity is secondary.

The sequential native-cache calibration and answer-form audits are in `V40_CALIBRATION_A3.md` and `V40_ANSWER_FORM_AUDIT.md`. They establish execution feasibility, not a channel role. Before formal development execution, an append-only design must freeze the answer readout and implement surface-only, irrelevant-history, same-prefix-mismatch, and shuffled-state controls. Validation and independent final must not tune this design.

Machine sources: `configs/channel_function_v40.yaml`, `data/v40/sample_pool_manifest_v40.json`, `results/v40/processed/design_Q_v40.json`, and `results/v40/processed/design_F_v40.json`.

---

<!-- V40_TASK_GROUNDED_RESULTS.md | SHA256 882ecb173a2e1592d1cd11a01d4e346b456bd33db5451d595782f60b447c1b2e -->

# V40 — Task-Grounded Results

Status: **calibration-only task outcomes recorded; no formal results**.

The sealed generator supplies four families: variable binding, finite state transition, long-context dependency, and Boolean XOR. Each state has recipient and donor prompts, one differing encoded token, and externally computed distinct correct answers. State is the independent statistical unit. These generated labels are not model outputs.

Calibration native answer recovery depends strongly on model and answer format; the machine-generated family breakdown is in `V40_ANSWER_FORM_AUDIT.md`. Formal task recovery, effect sizes, confidence intervals, and held-out family breakdown are pending. Representation similarity alone will not be treated as task recovery. No success criterion or semantic-storage claim has been adjudicated.

---

<!-- V40_TEMPORAL_PERSISTENCE.md | SHA256 79cd9222f43ba4beaba0b7d5c52d129abcc303309d9b4d3dafd45e73315b13c7 -->

# V40 — Temporal Persistence

Status: **single-state calibration pilots at selected delays; formal delay experiment not run**.

The frozen delay grid is 0, 1, 2, 4, 8, 16, and 32 shared neutral continuation tokens after the natural fork. ` neutral` passed tokenizer-level single-token checks in both models. One calibration state per model was run at delays 0, 1, and 32 under the sequential native-cache path; those pilots are not a population temporal profile. REC, Conv, KV, and REC+Conv effects and task-variable recovery remain to be measured formally at every delay.

No decay curve or channel half-life has been estimated. The analysis must first report empirical effect-versus-delay by task family; it must not impose exponential decay or infer a universal channel lifetime.

---

<!-- V40_CHANNEL_COMPARISON.md | SHA256 72ba7f9a1012702a2b431c479992a629b9eff7248ed3f444bbc74c38b4616418 -->

# V40 — Channel Comparison

Status: **calibration-only comparison; formal cross-model results not run**.

Qwen3.5-4B is primary for functional theory construction; Falcon-H1-1.5B-Base is secondary. Both share the same externally generated task states, but token IDs and native recurrent-layer layouts are model-specific. Each model has its own sealed single-token fork audit in `results/v40/processed/design_Q_v40.json` and `results/v40/processed/design_F_v40.json`.

Calibration shows the initial direct-answer readout is not equally interpretable across models; a leading-space answer audit improves Falcon's Boolean baseline but not the other families. See `V40_ANSWER_FORM_AUDIT.md`. This does not establish an architecture-specific channel role or global model deficit. A difference in mechanisms is permitted; identical channel roles are not assumed.

---

<!-- V40_TRAJECTORY_CAUSALITY.md | SHA256 20c5c1adca78274c1937e2e10652bc614ac01c4f7af8d1383010e49063847c2a -->

# V40 — Trajectory Causality

Status: **trajectory intervention not run**.

The planned primary test follows Qwen REC across early, middle, and late recurrent-block groups, replacing native donor states at selected checkpoints while all downstream computation runs naturally. Checkpoints must capture REC, Conv, hidden states, future logits, and externally defined task-variable recovery. No future output copying is allowed.

The precise block grouping, intervention schedule, and equality instrumentation must be sealed before formal outcomes. There is currently no V40 evidence that an early replacement alters how later states are used, or that a donor-like representation restores the task variable.

---

<!-- V40_COMPLETE_REPORT.md | SHA256 d5946146df517c72ad85eaed17fc30359f8a3599d7b479ffa0cff4eced434971 -->

# V40 — Complete Report (Calibration Checkpoint; Not Final)

V40 has been **started, not completed**. V39 remains unchanged. Four fresh task pools and the Qwen/Falcon one-token-fork designs are sealed. Append-only calibration amendments record why the initial prompt-collision selector was corrected and why Falcon's cached multi-token continuation was replaced by native single-token continuation. Both models completed the full calibration pool at delay zero with exact channel-write proofs and no invalid states. Separate single-state delay pilots and answer-format audits were also recorded.

The calibration answer-format audit finds the primary Qwen readout feasible, while Falcon has a baseline-qualified pair only in the Boolean family after a leading space. Consequently, broad Falcon task-role comparison is not yet interpretable. These are calibration diagnostics, not V40 hypothesis tests. The final question remains unanswered: channel content, causal lifetime, and trajectory transformation have not been formally established. Next, freeze a formal answer readout and controls, then run Qwen-led development, temporal and trajectory experiments. Validation and independent final remain unopened.

See `V40_CALIBRATION_A3.md` and `V40_ANSWER_FORM_AUDIT.md` for machine-generated calibration tables, `data/v40/sample_pool_manifest_v40.json` for sample provenance, and `results/v40/processed/startup_audit_v40.json` for the pairing audit. The required topic reports distinguish calibration from formal evidence.

---
