# J-State Closure and Cognitive Controller Experiment

This repository tests whether the Jacobian-lens J-space is approximately a
sufficient state for tested high-level model dynamics, a broadcast bus
whose future contents depend on computation outside measured J, or part of a compact
recurrent augmented state. It does **not** test or make claims about
consciousness.

The implementation uses Anthropic's `jacobian-lens` reference code and pins all
model, lens, and source revisions. Layer-depth closure and autoregressive
token-time closure are distinct analyses and are never treated as equivalent.

## Measurement gate

The original Phase 0 and its failed `phase0_gate.json` remain reproducible.
Protocol v2 corrects best-across-layer item-weighted pass@k, makes position 16 a
sensitivity rather than an exclusion, and expands order-of-operations targets.
It was frozen before a non-overlapping holdout was evaluated. The fresh v2
measurement gate passed, but the subsequent strict per-layer clamp calibration
found no closure-eligible layer. All formal downstream runners therefore stop
before model loading. This is a measurement/state-construction failure, not
evidence for H1, H2, or H3.

Exploratory protocol v3 does not alter that result. It first audits the geometry
of the unnormalized dense map `A_l = W_U J_l`, its centered map, and the local
Jacobian of the normalized dense profile. `V3-Dense` and `V3-Sparse` are separate
operational state definitions with separate equality tests. Behavioral closure
is disabled until a byte-frozen v3 calibration authorizes at least four ordered
layers while retaining a natural displacement of at least 0.20.
The formal v3 calibration is now complete: it saved 12,600 candidates and 4,527
formal-valid rows, but no protocol had more than one eligible layer. Behavioral
closure therefore remains gated and the required low-dimensional search is the
next authorized empirical stage. That screen has also completed: none of the
tested states at dimension at most 512 closed the preregistered 80% of the
J-only-to-remainder-oracle prediction gap (best 59.36%), so no compact state was
authorized and token-time/controller execution remains gated.

## Environment

The reference environment is Python 3.12.2, PyTorch 2.10.0+cu128, and
transformers 5.12.1. On the execution host, both RTX 4090 D GPUs passed a CUDA
matmul smoke test. Install in a clean environment:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

Large Hugging Face artifacts are downloaded into the normal cache or an
external directory and are verified against `artifacts/MANIFEST.json`; they are
never committed.

`requirements.lock` is generated with `uv pip compile constraints.txt
--python-version 3.12`. `constraints.txt` is the human-reviewed list of direct
pins, while the lock records the resolved transitive CUDA/Python dependency
graph.

## Commands

```bash
pytest -q
scripts/run_validation.sh
python -m jclosure.experiments.prepare_phase0_v2
scripts/run_validation_v2.sh
python -m jclosure.experiments.phase0_audit
python -m jclosure.experiments.freeze_phase0_v2
scripts/run_layer_calibration.sh
PYTHONPATH=src python scripts/check_v2_hashes.py
scripts/run_geometry_v3.sh
scripts/build_geometry_report.sh
scripts/run_clamp_v3_calibration.sh
scripts/freeze_phase3_v3.sh
scripts/run_closure_v3_pilot.sh
scripts/run_closure_v3_confirm.sh
scripts/run_mediation_v3.sh
scripts/run_dictionary_v3.sh
scripts/run_lowdim_search.sh
scripts/run_closure_pilot.sh
scripts/run_closure_confirm.sh
scripts/run_dictionary_sensitivity.sh
scripts/run_collision_search.sh
scripts/run_memory_order.sh
scripts/run_token_time.sh
scripts/run_distillation.sh
scripts/build_report.sh
```

## Persistent-channel attribution protocol v7

Protocol v7 keeps all v6 artifacts byte-guarded and decomposes Qwen3.5's hybrid
cache into full-attention K/V tensors, Gated DeltaNet recurrent matrices, and
short-convolution states. It first gates on exact save/restore one-token
continuation, then runs paired clean/KV-only/REC-only/full cache chimeras with a
shared teacher-forced continuation. Architecture-aligned compression is gated
on stable persistent-channel attribution; no generic recurrent controller is
trained in this protocol.

```bash
scripts/run_persistent_channels_v7.sh schema
scripts/run_persistent_channels_v7.sh restore
scripts/run_persistent_channels_v7.sh attribution
scripts/run_persistent_channels_v7.sh analyze
```

The architecture-channel tensor capture is followed by an additive corrective
analysis because the immutable first v7 endpoint archive serialized the final
`full` trajectory under every condition label. Attribution scalars and curves
were computed before that serialization and are unaffected. The corrective
protocol leaves the defective archive in place, uses the frozen v6 step-1
clean/intervened J endpoints, and refuses to run compression unless the fixed
raw-channel ceiling passes:

```bash
scripts/run_arch_compression_v7_corrective.sh freeze
scripts/run_arch_compression_v7_corrective.sh ceiling
scripts/run_arch_compression_v7_corrective.sh run
scripts/run_arch_compression_v7_corrective.sh analyze
```

## Peripheral foundations protocol v6

Protocol v6 is additive and leaves Phase 0 and v1--v5 files byte-guarded. It
repairs the teacher intervention endpoint by storing float32 residuals,
measured-J profiles, deltas, operational remainders, and logits. The causal
endpoint is the immediate same-forward layer-24 write after a layer-23
final-token intervention; the older next-token/layer-23 endpoint is retained
only as a forensic sensitivity because causal masking makes it structurally
insensitive unless the emitted token changes.

Persistent-null arms execute the same later layers, position scopes, projector,
and hook schedule as persistent restoration but start from the clean layer-23
state. Reports subtract this clean-state artifact before computing descriptive
mediation. The null does not measure state-dependent projector distortion.

The strong ceiling compares J-only, J-history, and full operational remainder
linear/gated/residual/attention predictors under a multitask next-J, causal
direction, top-coordinate, semantic-action, and output-sign objective.
Architecture selection uses ordinary validation only; causal-test never enters
selection. A gated compact sweep over 16/32/64/128/256/512D PCA, predictive,
and nonlinear bottlenecks runs only after a ceiling endpoint passes. No
recurrent controller is trained in v6.

```bash
scripts/run_peripheral_v6.sh freeze --run-suffix protocol-freeze-r3
CUDA_VISIBLE_DEVICES=0 scripts/run_peripheral_v6.sh causal --run-suffix full-66
scripts/run_peripheral_v6.sh merge --run-suffix merge-full66
CUDA_VISIBLE_DEVICES=1 scripts/run_peripheral_v6.sh ceiling --run-suffix leakfree-strong
CUDA_VISIBLE_DEVICES=1 scripts/run_peripheral_v6.sh compact --run-suffix compact-sweep-matched
scripts/build_report_v6.sh
```

## Peripheral computation protocol v5

Protocol v5 keeps every v1–v4 result immutable and holds the measured-J state
fixed at the layer-23 normalized 4,096-concept profile. It asks whether a
train-fitted measured-J remainder contains information about the next J write,
and whether that information can be encoded in a 16–512D peripheral state.
The full-remainder ceiling compares J-only, J-history, and three nonlinear
remainder-aware predictors before compact-state results are interpreted.

PCA, predictive-linear, and nonlinear bottlenecks are selected on validation
data and evaluated on the frozen rollout-test split. A compact candidate must
close at least 80% of the J-only-to-full-remainder gap, leave at most 0.002
conditional residual cosine gain, and pass teacher/student causal direction
and output-sign fidelity. Recurrent training is gated to authorized candidates
of at most 128 dimensions. Autonomous recurrence feeds back predicted J,
peripheral state, and action; it never reads a future teacher state or action.

An independent family-wise H2 arm uses fresh, program-disjoint Boolean,
state-transition, modular-arithmetic, graph, and binding tasks. It estimates
same-J/changed-hidden effects first and only runs persistent restoration for
families whose lower JS confidence bound exceeds the frozen noise floor.

```bash
scripts/run_peripheral_v5.sh --stage prepare --run-suffix layer23-derived
scripts/run_peripheral_v5.sh --stage freeze --run-suffix preregistered
scripts/run_peripheral_v5.sh --stage ceiling --run-suffix full-reference
scripts/run_h2_replication_v5.sh --stage single --run-suffix family-replication
scripts/run_h2_replication_v5.sh --stage merge --run-suffix merge
scripts/run_peripheral_v5.sh --stage sweep --run-suffix compact-sweep
scripts/run_peripheral_v5.sh --stage conditional --run-suffix conditional
scripts/run_peripheral_v5.sh --stage fidelity --run-suffix fidelity
scripts/run_peripheral_v5.sh --stage recurrent --run-suffix recurrent
scripts/build_report_v5.sh
scripts/build_postrun_report_v5.sh
```

All commands accept `CONFIG=...` and additional arguments through `ARGS`. Raw
records are append-only JSONL/Parquet partitions. Processed tables and figures
are derived from those records.

## Status

No model-scale result is claimed merely because the code exists. See
`reports/PHASE0_VALIDATION.md`, `reports/PHASE0_PROTOCOL_AUDIT.md`,
`reports/PHASE0_V2_CONFIRMATORY.md`, and `reports/FINAL_REPORT.md` for the
actual execution status and evidence boundary. The strongest warranted result is
classification D: the Phase 0 readout passed, but neither v2 nor exploratory-v3
state construction produced the four eligible layers required for behavioral
closure, so H1/H2/H3 and controller hypotheses remain unresolved.

For offline model execution, set `JCLOSURE_MODEL_DIR` to a snapshot containing
the pinned `artifact_manifest.json` and `JCLOSURE_ARTIFACT_DIR` to the verified
lens cache. The loaders hash-check every declared local artifact before use.
Failed runs remain as manifests and are never silently retried.

## Geometry-v3 execution notes

The formal activation bank contains 512 distinct prompt hashes: 64 states from
each of eight task families, split into 256 geometry-fit and 256 geometry-audit
states. Spectrum execution is sharded deterministically by layer across the two
GPUs. Each dictionary/layer combination stores complete local spectra for 16
fixed audit states and analytic/interlacing diagnostics for all 256 audit
states; smoke-only Parquet files are excluded from formal report status.

Pareto execution uses two hash-selected audit anchors per task family. It writes
one immutable part per dictionary/layer and an atomic progress manifest before
merging shard outputs. Preflight outputs contain `preflight` in the filename and
are excluded from formal figures and conclusions. The 0.20 displacement gate is
unchanged; the reduced Pareto anchor count limits geometric generalization and
is reported as an exploratory sampling choice, not a threshold adjustment.
Hard-constrained optimization enforces the train-fit naturality envelope during
backtracking. The subsequent 200-trial calibration batch is independently
balanced at 25 anchors per task family and uses IDs that remain paired across
dictionary sizes. Tangent construction uses an eight-FP32-ulp analytic margin
plus a measured adaptive correction with a 64-epsilon minimum increment so the
actual post-retraction chord cannot remain below the unchanged nominal 0.20
threshold; achieved displacement is always stored unrounded. Fixed-margin-only
retry records are retained as failed provenance.

The formal calibration replayed 200 anchors at each of seven layers, three
dictionary sizes, and three methods (12,600 rows total). All candidates were
finite, no activation explosion occurred, and zero/identity/determinism/cleanup
controls passed. Only M4096 Dense-optimized/L23 and M8192/M16384 Dense/L29 were
eligible; no state-definition/dictionary protocol had the required four ordered
layers. The first merge attempt is retained as a failed manifest because the
per-shard digest included the assigned CUDA device. The corrected merge excludes
only `model.device` from shard equivalence, records both original digests, and
still rejects every other config difference; it reused the immutable shard
Parquet files and did not recompute trials.

The low-dimensional screen used 1,536 fit and 1,536 audit layer transitions.
Last-state persistence reached median next-state cosine 0.965006 and the
remainder-aware oracle reached 0.990099. The best tested compact candidates were
512-D dense-profile PCA (gap closed 0.593644) and a 512-D predictive bottleneck
(0.593617); the 512-D constrained encoder reached 0.501637 and sparse active
atoms reached 0.388009. These observational prediction metrics do not authorize
causal fidelity or an H3/controller claim.

If calibration authorizes behavioral execution, the v3 pilot and confirmation
launchers shard prompts across both GPUs. Sample targets refer to valid base
trials per protocol, task family, and perturbation source; controls, strengths,
and clamp modes are paired records rather than separately inflated cells. The
final-position arm aligns each sequence at its own final token, while the
all-non-padding arm requires a length- and template-matched donor.
Supporting L0 layers outside the geometry bank's 23–29 range are re-recorded
from the saved input IDs under the frozen model; a run-local manifest records
their prompt/layer shapes and hashes, while the original geometry bank remains
unchanged.

## Corrective protocol v3.1 and compact-memory arm

Protocol v3.1 is additive and leaves every v1/v2/v3 configuration, freeze,
record, and dedicated report under SHA-256 guards. The corrective causal arm
separates the large initial intervention from later, potentially tiny
restoration corrections. The direct-L1 intervention still has to move at least
0.20 natural-difference units; restoration never inherits that lower bound.
Isolated and runtime-matched restoration chains are calibrated before a
behavioral freeze can be created.

The primary v3.1 arithmetic domain is a balanced integer-parity task. It was
fixed before causal outcomes because Qwen3.5-4B did not yield the declared 200
teacher-correct calibration examples on the earlier multi-operation candidate
domain with explicit thinking disabled. Superseded smokes and their failure
reasons remain in `results/v3_1/raw/`.

The independent compact-memory arm uses token time rather than layer depth. It
extracts greedy teacher traces for iterated modular arithmetic and synthetic
finite-state-machine traversal, fits every representation on train only, and
evaluates Markov, true-history, and persistent-GRU controllers by autonomous
feedback from `Z0`. Remainder-aware references are explicitly separated into
two teacher-current one-step endpoints (linear PCA-128 and nonlinear full
remainder) and one PCA-512 recurrent endpoint that reads only `(Z0,R0)` before
feeding back its own predicted J/remainder state. Only the recurrent endpoint
is an autonomous comparison.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export JCLOSURE_MODEL_DIR=/data/CSK/J-space-project/models/Qwen3.5-4B-851bf6e
export JCLOSURE_ARTIFACT_DIR=/data/CSK/J-space-project/.jclosure-artifacts

python -m jclosure.experiments.prepare_v3_1
scripts/run_calibration_v3_1.sh
scripts/freeze_v3_1.sh closure
scripts/run_closure_v3_1_pilot.sh
scripts/run_closure_v3_1_confirm.sh
scripts/freeze_v3_1.sh memory
scripts/run_compact_memory_v3_1.sh
scripts/run_compact_memory_references_v3_1.sh
scripts/build_report_v3_1.sh
```

## Corrective causal protocol v3.2

Protocol v3.2 is additive: no v3.1 source, threshold, freeze, raw record, or
dedicated report is rewritten. It makes the primary intervention scope and the
later restoration scope independent. `single`, `persistent_final`, and
`persistent_all` share one final-token direct-L1 candidate; their only causal
difference is whether later layers restore no positions, the final position,
or every non-padding position. Hook records include layer, position, and
operation type.

Restoration calibration compares the existing local and hard-constrained
optimized projectors. Layer eligibility is conditional on a valid initial
intervention and reports both per-layer and complete-chain bootstrap intervals.
The all-position *initial* arm is not part of the primary authorization gate.

Compact-memory v3.2 consumes the already collected six v3.1 trace shards. It
does not regenerate trajectories. A canonical audit checks identities, split
and program leakage, record fields, tensor hashes, tensor shapes, and finite
values before representation screening. Student agreement with teacher actions
is always reported separately from ground-truth task accuracy.

```bash
scripts/prepare_v3_2.sh
scripts/run_calibration_v3_2.sh
python -m jclosure.experiments.freeze_v3_2 --kind closure
scripts/run_closure_v3_2_pilot.sh
scripts/run_closure_v3_2_confirm.sh
STAGE=merge-audit scripts/run_compact_memory_v3_2.sh
STAGE=screen scripts/run_compact_memory_v3_2.sh
STAGE=train scripts/run_compact_memory_v3_2.sh
scripts/build_report_v3_2.sh
```

## Teacher-competent predictive-state protocol v4

Protocol v4 is additive and does not rerun or rewrite Phase 0, the geometry
audit, or earlier closure records. It first calibrates five machine-scored
program families at horizons 4/8/16/32, freezes disjoint train, validation,
causal-test, and rollout-test domains, and extracts token-time traces only from
teacher trajectories that are completely correct.

The causal runner intervenes once at block-output layer 23 on the final prompt
position and then permits ordinary KV-cached autoregressive continuation. Six
conditions share each anchor/donor pair: clean, identity, norm-matched random,
intended-answer J direction, full donor difference, and a dense measured-J
preserving perturbation. Persistent restoration is deliberately downstream of
the single-arm noise gate.

The compact-state screen compares 64/128/256/512D PCA, linear predictive
bottleneck, sparse-J coordinates, and nonlinear learned predictive encoders.
Its loss combines current reconstruction, future prediction, semantic action,
and validated intervention-delta retention. If no representation clears every
retention gate, temporal training may still run as explicitly exploratory using
a deterministic maximin-retention fallback; this never authorizes a compact
state claim.

```bash
scripts/run_teacher_v4.sh --stage calibrate --run-suffix calibration
scripts/run_teacher_v4.sh --stage freeze --run-suffix freeze
scripts/run_teacher_v4.sh --stage formal --run-suffix formal
scripts/run_predictive_state_v4.sh traces --domain all --run-suffix traces
scripts/run_single_arm_v4.sh --stage bank --run-suffix bank
scripts/run_single_arm_v4.sh --stage run --run-suffix pilot
scripts/run_single_arm_v4.sh --stage merge --run-suffix merge
scripts/run_predictive_state_v4.sh screen --run-suffix screen
scripts/run_controllers_v4.sh
scripts/run_predictive_state_v4.sh reference --run-suffix references
scripts/build_report.sh
```
