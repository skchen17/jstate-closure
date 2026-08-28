#!/usr/bin/env bash
set -euo pipefail
if [[ -d results/v3/raw ]] && find results/v3/raw -path '*/map_spectra-*.parquet' -print -quit | grep -q .; then
  python -m jclosure.reporting_v3 --config "${CONFIG:-configs/geometry_v3.yaml}" ${ARGS:-}
elif [[ -f results/processed/phase0_v2_gate.json ]]; then
  CONFIG="${CONFIG:-configs/confirm_v2.yaml}"
  python -m jclosure.reporting_v2 --config "$CONFIG" ${ARGS:-}
else
  CONFIG="${CONFIG:-configs/confirm.yaml}"
  python -m jclosure.reporting --config "$CONFIG" ${ARGS:-}
fi
