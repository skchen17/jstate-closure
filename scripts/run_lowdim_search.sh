#!/usr/bin/env bash
set -euo pipefail
timeout 24h python -m jclosure.experiments.lowdim_search \
  --config "${CONFIG:-configs/geometry_v3.yaml}" "$@"
