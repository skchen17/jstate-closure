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
feedback from `Z0`.

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
scripts/build_report_v3_1.sh
```
