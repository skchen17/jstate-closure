#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
python -m ruff check .
python -m mypy \
  src/jclosure/geometry.py \
  src/jclosure/clamp_v3.py \
  src/jclosure/protocol_v3.py \
  src/jclosure/records.py
python -m mypy --ignore-missing-imports \
  src/jclosure/experiments/geometry_v3.py \
  src/jclosure/experiments/clamp_v3_calibration.py \
  src/jclosure/experiments/closure_v3.py \
  src/jclosure/experiments/lowdim_search.py \
  src/jclosure/reporting_v3.py
python -m mypy --follow-imports=skip --ignore-missing-imports \
  src/jclosure/records_v3_1.py \
  src/jclosure/clamp_v3_1.py \
  src/jclosure/datasets_v3_1.py \
  src/jclosure/runtime_v3_1.py \
  src/jclosure/protocol_v3_1.py \
  src/jclosure/statistics_v3_1.py \
  src/jclosure/compact_memory_v3_1.py
python -m mypy --ignore-missing-imports \
  src/jclosure/protocol_v4.py \
  src/jclosure/records_v4.py \
  src/jclosure/single_arm_v4.py \
  src/jclosure/predictive_state_v4.py \
  src/jclosure/reporting_v4.py \
  src/jclosure/experiments/causal_single_v4.py \
  src/jclosure/experiments/mediation_v4.py \
  src/jclosure/experiments/traces_v4.py \
  src/jclosure/experiments/predictive_state_v4.py \
  src/jclosure/experiments/controllers_v4.py \
  src/jclosure/experiments/references_v4.py \
  tests/test_v4.py
PYTHONPATH=src python scripts/check_v2_hashes.py
PYTHONPATH=src python scripts/check_v3_immutable.py
python scripts/check_repository_artifacts.py
git diff --check
