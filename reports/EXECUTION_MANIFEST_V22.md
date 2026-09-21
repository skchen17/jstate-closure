# Execution Manifest — V22

Exact primary commands:

```text
PYTHONPATH=src python -m jclosure.protocol_v22 freeze
PYTHONPATH=src python -m jclosure.experiments.input_geometry_v22 prepare
PYTHONPATH=src python -m jclosure.experiments.input_geometry_v22 run
PYTHONPATH=src python -m jclosure.experiments.action_pool_v22 prepare
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m jclosure.experiments.action_pool_v22 calibrate
PYTHONPATH=src python -m jclosure.experiments.action_selection_v22
PYTHONPATH=src python -m jclosure.experiments.expanded_bank_v22 prepare
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m jclosure.experiments.expanded_bank_v22 run --role development
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m jclosure.experiments.expanded_bank_v22 run --role validation
PYTHONPATH=src python -m jclosure.experiments.expanded_bank_v22 summarize
PYTHONPATH=src python -m jclosure.experiments.analyze_v22
PYTHONPATH=src python -m jclosure.experiments.scale_composition_v22 prepare
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m jclosure.experiments.scale_composition_v22 run --role development
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m jclosure.experiments.scale_composition_v22 run --role validation
PYTHONPATH=src python -m jclosure.experiments.scale_composition_v22 analyze
PYTHONPATH=src python scripts/finalize_v22.py
PYTHONPATH=src pytest -q tests/test_v22.py
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/freeze_v22_final.py
git add ... && git commit -m 'Complete V22 causal action manifold study'
git push origin main
```

Changed files before commit:

```text
artifacts/causal_action_manifold_v22.freeze.json
artifacts/causal_action_manifold_v22_action_pool_and_partitions.freeze.json
artifacts/causal_action_manifold_v22_action_reliability.freeze.json
artifacts/causal_action_manifold_v22_action_selection.freeze.json
artifacts/causal_action_manifold_v22_analysis_results.freeze.json
artifacts/causal_action_manifold_v22_expanded_bank_design.freeze.json
artifacts/causal_action_manifold_v22_final.freeze.json
artifacts/causal_action_manifold_v22_input_metric.freeze.json
artifacts/causal_action_manifold_v22_scale_composition_design.freeze.json
configs/causal_action_manifold_v22.yaml
reports/ACTION_COMPOSITION_V22.md
reports/ACTION_DATA_SCALING_V22.md
reports/ACTION_EXPERIMENT_DESIGN_V22.md
reports/ACTION_POOL_CALIBRATION_V22.md
reports/ACTION_SCALE_LAW_V22.md
reports/CAUSAL_ACTION_COORDINATES_V22.md
reports/CAUSAL_COVERAGE_ERROR_V22.md
reports/COMPACT_OPERATOR_REOPEN_V22.md
reports/EVEN_ODD_ACTION_GEOMETRY_V22.md
reports/EXECUTION_MANIFEST_V22.md
reports/FINAL_REPORT.md
reports/INDEPENDENT_ACTION_FINAL_V22.md
reports/INPUT_SIDE_CAUSAL_GEOMETRY_V22.md
reports/LOW_RANK_OPERATOR_IDENTIFICATION_V22.md
reports/STATE_CONDITIONED_ACTION_CHART_V22.md
reports/STRICT_INTERFACE_AUDIT_V22.md
reports/V22_COMPLETE_REPORT.md
reports/V22_SCIENTIFIC_ANSWERS_V22.md
results/v22/
results/v22/processed/v22_verification.json
scripts/finalize_v22.py
scripts/freeze_v22_final.py
scripts/test_audit_v22.py
src/jclosure/experiments/action_pool_v22.py
src/jclosure/experiments/action_selection_v22.py
src/jclosure/experiments/analyze_v22.py
src/jclosure/experiments/expanded_bank_v22.py
src/jclosure/experiments/input_geometry_v22.py
src/jclosure/experiments/scale_composition_v22.py
src/jclosure/protocol_v22.py
tests/test_v22.py
```

GPU measurements used `CUDA_VISIBLE_DEVICES=1`, `HF_HOME=/data/CSK/J-space-project/.hf-cache`, and the repository's Python 3.12 environment. Closed-form model outputs have no neural weight artifact; hashes are in `model_registry_v22.json`.
