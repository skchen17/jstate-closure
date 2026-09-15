#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: run_compression_v8_1.sh <freeze|analyze> [extra args]}"
shift
python -m jclosure.experiments.compress_persistent_v8_1 \
  --config configs/persistent_state_v8.yaml \
  --stage "${stage}" \
  "$@"
