# V38 — Frozen Starting Point

Parent commit `ef075e85025eb634b0a02fecadc16c8a4a6811e9`; V1–V37 reports/results unchanged and V37 outcomes excluded from V38 formal decisions. New pools contain calibration/development/validation/final = 20/80/40/40 states per model, five families, all cross-role program/prompt-disjoint and historically disjoint. Validation/final were sealed before intervention responses. Pool hashes and model/checkpoint/tokenizer hashes are in `artifacts/trajectory_composition_v38.freeze.json` and `data/v38/sample_pool_manifest_v38.json`.

Important distribution shift: modular arithmetic uses horizon 2 in calibration/development and horizon 3 in validation/final, after a sealed, outcome-blind amendment. This limits same-distribution interpretation. V18 training leftovers were not used in formal roles.
