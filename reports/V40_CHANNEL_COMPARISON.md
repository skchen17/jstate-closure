# V40 — Channel Comparison

Status: **cross-model outcomes not run**.

Qwen3.5-4B is primary for functional theory construction; Falcon-H1-1.5B-Base is secondary. Both share the same externally generated task states, but token IDs and native recurrent-layer layouts are model-specific. Each model has its own sealed single-token fork audit in `results/v40/processed/design_Q_v40.json` and `results/v40/processed/design_F_v40.json`.

No architecture-specific functional difference has yet been measured. A difference in mechanisms is permitted; identical channel roles are not assumed.
