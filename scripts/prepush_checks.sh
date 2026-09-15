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
python -m mypy --ignore-missing-imports \
  src/jclosure/datasets_v5.py \
  src/jclosure/protocol_v5.py \
  src/jclosure/records_v5.py \
  src/jclosure/peripheral_v5.py \
  src/jclosure/experiments/peripheral_v5.py \
  src/jclosure/experiments/h2_replication_v5.py \
  src/jclosure/reporting_v5.py \
  src/jclosure/reporting_postrun_v5.py \
  tests/test_v5.py
python -m mypy --ignore-missing-imports \
  src/jclosure/protocol_v6.py \
  src/jclosure/records_v6.py \
  src/jclosure/peripheral_v6.py \
  src/jclosure/experiments/causal_endpoint_v6.py \
  src/jclosure/experiments/peripheral_ceiling_v6.py \
  src/jclosure/reporting_v6.py \
  tests/test_v6.py
python -m mypy --ignore-missing-imports \
  src/jclosure/cache_v7.py \
  src/jclosure/protocol_v7.py \
  src/jclosure/protocol_v7_corrective.py \
  src/jclosure/protocol_v7_stage2.py \
  src/jclosure/records_v7.py \
  src/jclosure/arch_compression_v7.py \
  src/jclosure/experiments/persistent_channels_v7.py \
  src/jclosure/experiments/localize_channels_v7.py \
  src/jclosure/experiments/localization_analysis_v7.py \
  src/jclosure/experiments/arch_compression_v7.py \
  src/jclosure/experiments/arch_compression_v7_corrective.py \
  src/jclosure/experiments/report_v7.py \
  src/jclosure/reporting_v7.py \
  tests/test_v7.py \
  tests/test_v7_corrective.py
PYTHONPATH=src python scripts/check_v2_hashes.py
PYTHONPATH=src python scripts/check_v3_immutable.py
PYTHONPATH=src python scripts/check_v4_immutable.py
PYTHONPATH=src python - <<'PY'
from pathlib import Path

from jclosure.config import load_config
from jclosure.protocol_v6 import verify_freeze, verify_v5_guard

root = Path(".").resolve()
verify_v5_guard(root)
verify_freeze(root, load_config(root / "configs/peripheral_v6.yaml"))
print("v5/v6 immutable guards passed")
PY
PYTHONPATH=src python - <<'PY'
from pathlib import Path

from jclosure.config import load_config
from jclosure.protocol_v7 import verify_freeze, verify_v6_guard
from jclosure.protocol_v7_corrective import verify_freeze as verify_corrective
from jclosure.protocol_v7_stage2 import verify_stage_freeze

root = Path(".").resolve()
verify_v6_guard(root)
verify_freeze(root, load_config(root / "configs/persistent_channels_v7.yaml"))
config = load_config(root / "configs/persistent_channels_v7.yaml")
verify_stage_freeze(root, config, "localization")
verify_stage_freeze(root, config, "compression")
verify_corrective(
    root, load_config(root / "configs/persistent_channels_v7_corrective.yaml")
)
print("v6/v7/v7.1 immutable guards passed")
PY
python scripts/check_repository_artifacts.py
git diff --check
