#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/compact_memory_v3_1.yaml}"
STAGE="${STAGE:-all}"
GROUP="${JCLOSURE_SHARD_GROUP_ID:-compact-memory-v3-1-$(date -u +%Y%m%dT%H%M%SZ)}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then LIMIT_ARGS=(--limit "$LIMIT"); fi
if [[ -z "${JCLOSURE_GIT_WORKTREE_SNAPSHOT:-}" ]]; then
  JCLOSURE_GIT_WORKTREE_SNAPSHOT="$(python -c 'import json; from jclosure.provenance import git_worktree_snapshot; print(json.dumps(git_worktree_snapshot("."), sort_keys=True))')"
  export JCLOSURE_GIT_WORKTREE_SNAPSHOT
fi
export JCLOSURE_SHARD_GROUP_ID="$GROUP"

if [[ "$STAGE" == "traces" || "$STAGE" == "all" ]]; then
  for SPLIT in train validation test; do
    timeout 72h python -m jclosure.experiments.compact_memory_v3_1 --config "$CONFIG" --stage traces --split "$SPLIT" --device 0 --shard-index 0 --shard-count 2 --shard-group-id "$GROUP" --run-suffix "$SPLIT-000" "${LIMIT_ARGS[@]}" & PID0=$!
    timeout 72h python -m jclosure.experiments.compact_memory_v3_1 --config "$CONFIG" --stage traces --split "$SPLIT" --device 1 --shard-index 1 --shard-count 2 --shard-group-id "$GROUP" --run-suffix "$SPLIT-001" "${LIMIT_ARGS[@]}" & PID1=$!
    trap 'kill -INT "$PID0" "$PID1" 2>/dev/null || true' INT TERM
    set +e; wait "$PID0"; S0=$?; wait "$PID1"; S1=$?; set -e
    if (( S0 != 0 || S1 != 0 )); then echo "memory trace shards failed for $SPLIT: $S0/$S1" >&2; exit 1; fi
  done
  timeout 72h python -m jclosure.experiments.compact_memory_v3_1 --config "$CONFIG" --stage merge-traces --shard-count 2 --shard-group-id "$GROUP"
fi
if [[ "$STAGE" == "traces" ]]; then exit 0; fi
if [[ "$STAGE" == "screen" || "$STAGE" == "all" ]]; then
  timeout 72h python -m jclosure.experiments.compact_memory_v3_1 --config "$CONFIG" --stage screen --device 0
fi
if [[ "$STAGE" == "screen" ]]; then exit 0; fi

if ! jq -e '.temporal_training_authorized == true' results/v3_1/processed/compact_memory_representation_screen.json >/dev/null; then
  echo "Temporal training gated by representation screen" >&2
  exit 0
fi
for DIMENSION in 64 128 256 512; do
  for SEED in 20260828 20260829 20260830; do
    timeout 72h python -m jclosure.experiments.compact_memory_v3_1 --config "$CONFIG" --stage train --device 0 --family markov --dimension "$DIMENSION" --controller-seed "$SEED"
    for HISTORY in 1 2 4 8 16; do
      timeout 72h python -m jclosure.experiments.compact_memory_v3_1 --config "$CONFIG" --stage train --device 0 --family history --dimension "$DIMENSION" --history "$HISTORY" --controller-seed "$SEED"
    done
    for MEMORY in 16 32 64 128 256; do
      timeout 72h python -m jclosure.experiments.compact_memory_v3_1 --config "$CONFIG" --stage train --device 1 --family gru --dimension "$DIMENSION" --memory-dimension "$MEMORY" --controller-seed "$SEED"
    done
  done
done
