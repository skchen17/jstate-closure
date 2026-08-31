#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/compact_memory_v3_2.yaml}"
STAGE="${STAGE:-references}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-20260828}"
timeout 72h python -m jclosure.experiments.compact_memory_references_v3_2 \
  --config "$CONFIG" --stage "$STAGE" --device "$DEVICE" \
  --controller-seed "$SEED"
