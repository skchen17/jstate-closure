#!/usr/bin/env bash
set -euo pipefail
timeout 24h python -m jclosure.experiments.closure_v3 \
  --config "${CONFIG:-configs/closure_v3_confirm.yaml}" "$@"
