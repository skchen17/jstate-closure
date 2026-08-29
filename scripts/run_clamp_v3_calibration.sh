#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/geometry_v3.yaml}"
BANK_MANIFEST="${BANK_MANIFEST:-$(find results/v3/raw -path '*/activation_bank_manifest.jsonl' -type f | sort | tail -n 1)}"
SHARD_GROUP_ID="${JCLOSURE_SHARD_GROUP_ID:-clamp-v3-$(date -u +%Y%m%dT%H%M%SZ)}"
LIMIT_ARGS=()
DRY_RUN=false
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi
for ARGUMENT in "$@"; do
  if [[ "$ARGUMENT" == "--dry-run" ]]; then
    DRY_RUN=true
  fi
done
if [[ -z "$BANK_MANIFEST" ]]; then
  echo "No geometry activation-bank manifest found" >&2
  exit 2
fi

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

timeout 12h python -m jclosure.experiments.clamp_v3_calibration \
  --config "$CONFIG" --bank-manifest "$BANK_MANIFEST" \
  --device 0 --shard-index 0 --shard-count 2 \
  --shard-group-id "$SHARD_GROUP_ID" --run-suffix clamp-shard-000 \
  "${LIMIT_ARGS[@]}" "$@" &
PID0=$!
timeout 12h python -m jclosure.experiments.clamp_v3_calibration \
  --config "$CONFIG" --bank-manifest "$BANK_MANIFEST" \
  --device 1 --shard-index 1 --shard-count 2 \
  --shard-group-id "$SHARD_GROUP_ID" --run-suffix clamp-shard-001 \
  "${LIMIT_ARGS[@]}" "$@" &
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
  echo "Clamp calibration shards failed: shard-000=$STATUS0 shard-001=$STATUS1" >&2
  exit 1
fi
if [[ "$DRY_RUN" == "true" ]]; then
  exit 0
fi

timeout 12h python -m jclosure.experiments.clamp_v3_calibration \
  --config "$CONFIG" --bank-manifest "$BANK_MANIFEST" \
  --merge-only --shard-count 2 --shard-group-id "$SHARD_GROUP_ID"
