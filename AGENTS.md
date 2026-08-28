# Repository instructions

This repository tests whether Jacobian-lens J-state is an approximately closed
dynamical state. It does not investigate or make claims about consciousness.

## Research-integrity gates

- Never invent results, sample counts, confidence intervals, or completed runs.
- `reports/PHASE0_VALIDATION.md` is the measurement gate. Later causal results
  may be interpreted only when its machine-readable gate artifact says `passed`.
- Preserve invalid and excluded trials in raw records with an explicit reason.
- Label natural-collision analyses observational; only intervention experiments
  support causal wording.
- Figures and numeric report tables must be generated from saved JSONL/Parquet
  records. Do not type measurements manually into plotting or report code.
- Do not interpret Qwen3.6-27B unless its lens passes independent shape,
  intermediate-readout, and positive-intervention validation.

## Engineering rules

- Keep model IDs, revisions, lens paths, layers, positions, seeds, thresholds,
  and sample sizes config-driven.
- Raw result partitions are immutable. Re-runs use a new `run_id`.
- Never commit model weights, lens tensors, activation banks, trace tensors,
  controller checkpoints, offload files, caches, or credentials. Commit hashes
  and manifests instead.
- All hooks must be context managed and removed after success or failure.
- Add or update CPU tests for changes to decomposition, interventions, clamping,
  metrics, schemas, or dataset generators.

