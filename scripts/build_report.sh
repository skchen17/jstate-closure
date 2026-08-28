#!/usr/bin/env bash
set -euo pipefail
if [[ -f results/processed/phase0_v2_gate.json ]]; then
  CONFIG="${CONFIG:-configs/confirm_v2.yaml}"
  python -m jclosure.reporting_v2 --config "$CONFIG" ${ARGS:-}
else
  CONFIG="${CONFIG:-configs/confirm.yaml}"
  python -m jclosure.reporting --config "$CONFIG" ${ARGS:-}
fi
