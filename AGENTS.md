# Repository instructions

This repository tests whether Jacobian-lens J-state is an approximately closed
dynamical state. It does not investigate or make claims about consciousness.

## Research-integrity gates

- Never invent results, sample counts, confidence intervals, or completed runs.
- The v1 `reports/PHASE0_VALIDATION.md` and `phase0_gate.json` remain immutable
  historical artifacts. New causal results may be interpreted only when the
  frozen v2 `phase0_v2_gate.json` says `passed` and per-layer calibration finds
  at least one closure-eligible layer.
- Preserve invalid and excluded trials in raw records with an explicit reason.
- Label natural-collision analyses observational; only intervention experiments
  support causal wording.
- Call finite-dictionary decompositions `measured-J component` and
  `measured-J remainder`; do not silently promote them to complete J/non-J space.
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
