#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/geometry_v3.yaml}"
STAGE="${STAGE:-all}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

if [[ "$STAGE" == "smoke" ]]; then
  timeout 8h python -m jclosure.experiments.geometry_v3 \
    --config "$CONFIG" --stage smoke "${LIMIT_ARGS[@]}" "$@"
  exit 0
fi

if [[ "$STAGE" == "bank" || "$STAGE" == "all" ]]; then
  timeout 8h python -m jclosure.experiments.geometry_v3 \
    --config "$CONFIG" --stage bank "${LIMIT_ARGS[@]}" "$@"
fi

BANK_MANIFEST="${BANK_MANIFEST:-$(find results/v3/raw -path '*/activation_bank_manifest.jsonl' -type f | sort | tail -n 1)}"
if [[ -z "$BANK_MANIFEST" ]]; then
  echo "No geometry activation-bank manifest found" >&2
  exit 2
fi

if [[ "$STAGE" == "spectrum" || "$STAGE" == "all" ]]; then
  timeout 8h python -m jclosure.experiments.geometry_v3 \
    --config "$CONFIG" --stage spectrum --bank-manifest "$BANK_MANIFEST" \
    --device 0 --shard-index 0 --shard-count 2 "${LIMIT_ARGS[@]}" "$@" &
  PID0=$!
  timeout 8h python -m jclosure.experiments.geometry_v3 \
    --config "$CONFIG" --stage spectrum --bank-manifest "$BANK_MANIFEST" \
    --device 1 --shard-index 1 --shard-count 2 "${LIMIT_ARGS[@]}" "$@" &
  PID1=$!
  wait "$PID0"
  wait "$PID1"
fi

if [[ "$STAGE" == "pareto" || "$STAGE" == "all" ]]; then
  timeout 12h python -m jclosure.experiments.geometry_v3 \
    --config "$CONFIG" --stage pareto --bank-manifest "$BANK_MANIFEST" \
    --device 0 "${LIMIT_ARGS[@]}" "$@"
fi
