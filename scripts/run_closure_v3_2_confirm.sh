#!/usr/bin/env bash
set -euo pipefail
jq -e '.instrumentation_gate.passed == true' results/v3_2/processed/closure_v3_2_pilot.json >/dev/null
CONFIG="${CONFIG:-configs/closure_v3_2.yaml}"
GROUP="${JCLOSURE_SHARD_GROUP_ID:-closure-v3-2-confirm-$(date -u +%Y%m%dT%H%M%SZ)}"
export JCLOSURE_SHARD_GROUP_ID="$GROUP"
timeout 24h python -m jclosure.experiments.closure_v3_2 --config "$CONFIG" --domain confirmation --device 0 --stage run --shard-index 0 --shard-count 2 --shard-group-id "$GROUP" --run-suffix confirm-000 & PID0=$!
timeout 24h python -m jclosure.experiments.closure_v3_2 --config "$CONFIG" --domain confirmation --device 1 --stage run --shard-index 1 --shard-count 2 --shard-group-id "$GROUP" --run-suffix confirm-001 & PID1=$!
trap 'kill -INT "$PID0" "$PID1" 2>/dev/null || true' INT TERM
set +e; wait "$PID0"; S0=$?; wait "$PID1"; S1=$?; set -e
if (( S0 != 0 || S1 != 0 )); then echo "v3.2 confirmation failed: $S0/$S1" >&2; exit 1; fi
timeout 24h python -m jclosure.experiments.closure_v3_2 --config "$CONFIG" --domain confirmation --stage merge --shard-count 2 --shard-group-id "$GROUP"
