#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/closure_v3_2.yaml}"
GROUP="${JCLOSURE_SHARD_GROUP_ID:-closure-v3-2-pilot-$(date -u +%Y%m%dT%H%M%SZ)}"
LIMIT_ARGS=(); if [[ -n "${LIMIT:-}" ]]; then LIMIT_ARGS=(--limit "$LIMIT"); fi
export JCLOSURE_SHARD_GROUP_ID="$GROUP"
timeout 24h python -m jclosure.experiments.closure_v3_2 --config "$CONFIG" --domain pilot --device 0 --stage run --shard-index 0 --shard-count 2 --shard-group-id "$GROUP" --run-suffix pilot-000 "${LIMIT_ARGS[@]}" & PID0=$!
timeout 24h python -m jclosure.experiments.closure_v3_2 --config "$CONFIG" --domain pilot --device 1 --stage run --shard-index 1 --shard-count 2 --shard-group-id "$GROUP" --run-suffix pilot-001 "${LIMIT_ARGS[@]}" & PID1=$!
trap 'kill -INT "$PID0" "$PID1" 2>/dev/null || true' INT TERM
set +e; wait "$PID0"; S0=$?; wait "$PID1"; S1=$?; set -e
if (( S0 != 0 || S1 != 0 )); then echo "v3.2 pilot failed: $S0/$S1" >&2; exit 1; fi
timeout 24h python -m jclosure.experiments.closure_v3_2 --config "$CONFIG" --domain pilot --stage merge --shard-count 2 --shard-group-id "$GROUP"
