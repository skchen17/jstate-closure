# J-State Closure and Cognitive Controller Experiment

This repository tests whether the Jacobian-lens J-space is approximately a
sufficient layer-depth state for high-level model dynamics, a broadcast bus
whose future contents depend on non-J computation, or part of a compact
recurrent augmented state. It does **not** test or make claims about
consciousness.

The implementation uses Anthropic's `jacobian-lens` reference code and pins all
model, lens, and source revisions. The primary state is the residual activation
at the final pre-answer token, and dynamical time is transformer layer depth.

## Measurement gate

Phase 0 must demonstrate hidden-intermediate readout and a positive-control J
intervention. Every later runner reads `results/processed/phase0_gate.json` and
refuses confirmatory interpretation if the gate is absent or failed. A failed
gate is a measurement failure, not evidence for any hypothesis.

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
scripts/run_closure_pilot.sh
scripts/run_closure_confirm.sh
scripts/run_collision_search.sh
scripts/run_memory_order.sh
scripts/run_distillation.sh
scripts/build_report.sh
```

All commands accept `CONFIG=...` and additional arguments through `ARGS`. Raw
records are append-only JSONL/Parquet partitions. Processed tables and figures
are derived from those records.

## Status

No model-scale result is claimed merely because the code exists. See
`reports/PHASE0_VALIDATION.md` and `reports/FINAL_REPORT.md` for the actual
execution status and evidence boundary.
