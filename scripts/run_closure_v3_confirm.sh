#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/closure_v3_confirm.yaml}"
SHARD_GROUP_ID="${JCLOSURE_SHARD_GROUP_ID:-closure-v3-confirm-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -z "${JCLOSURE_GIT_WORKTREE_SNAPSHOT:-}" ]]; then
  JCLOSURE_GIT_WORKTREE_SNAPSHOT="$(python -c 'import json; from jclosure.provenance import git_worktree_snapshot; print(json.dumps(git_worktree_snapshot("."), sort_keys=True))')"
  export JCLOSURE_GIT_WORKTREE_SNAPSHOT
fi
export JCLOSURE_SHARD_GROUP_ID="$SHARD_GROUP_ID"

PID0=""
PID1=""
stop_pair() {
  [[ -z "$PID0" ]] || kill -INT "$PID0" 2>/dev/null || true
  [[ -z "$PID1" ]] || kill -INT "$PID1" 2>/dev/null || true
  [[ -z "$PID0" ]] || wait "$PID0" 2>/dev/null || true
  [[ -z "$PID1" ]] || wait "$PID1" 2>/dev/null || true
  exit 130
}
trap stop_pair INT TERM

timeout 24h python -m jclosure.experiments.closure_v3 \
  --config "$CONFIG" --device 0 --shard-index 0 --shard-count 2 \
  --shard-group-id "$SHARD_GROUP_ID" \
  --run-suffix closure-confirm-shard-000 "$@" &
PID0=$!
timeout 24h python -m jclosure.experiments.closure_v3 \
  --config "$CONFIG" --device 1 --shard-index 1 --shard-count 2 \
  --shard-group-id "$SHARD_GROUP_ID" \
  --run-suffix closure-confirm-shard-001 "$@" &
PID1=$!

set +e
wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?
set -e
PID0=""
PID1=""
if (( STATUS0 != 0 || STATUS1 != 0 )); then
  echo "Closure confirmation shards failed: shard-000=$STATUS0 shard-001=$STATUS1" >&2
  exit 1
fi
