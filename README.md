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
classification D: the Phase 0 readout passed, but strict clamp calibration failed
at every candidate layer, so closure/controller hypotheses remain unresolved.

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
dictionary sizes.
