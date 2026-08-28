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
PYTHONPATH=src python scripts/check_v2_hashes.py
python scripts/check_repository_artifacts.py
git diff --check
