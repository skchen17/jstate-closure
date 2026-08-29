#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/geometry_v3.yaml}"
STAGE="${STAGE:-all}"
LIMIT_ARGS=()
PID0=""
PID1=""
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

# Capture repository state before a shard creates an untracked run directory.
if [[ -z "${JCLOSURE_GIT_WORKTREE_SNAPSHOT:-}" ]]; then
  JCLOSURE_GIT_WORKTREE_SNAPSHOT="$(python -c 'import json; from jclosure.provenance import git_worktree_snapshot; print(json.dumps(git_worktree_snapshot("."), sort_keys=True))')"
  export JCLOSURE_GIT_WORKTREE_SNAPSHOT
fi

stop_pair() {
  [[ -z "$PID0" ]] || kill -INT "$PID0" 2>/dev/null || true
  [[ -z "$PID1" ]] || kill -INT "$PID1" 2>/dev/null || true
  [[ -z "$PID0" ]] || wait "$PID0" 2>/dev/null || true
  [[ -z "$PID1" ]] || wait "$PID1" 2>/dev/null || true
  exit 130
}

wait_pair() {
  local status0 status1
  set +e
  wait "$PID0"
  status0=$?
  wait "$PID1"
  status1=$?
  set -e
  PID0=""
  PID1=""
  if (( status0 != 0 || status1 != 0 )); then
    echo "Geometry shards failed: shard-000=$status0 shard-001=$status1" >&2
    return 1
  fi
}

trap stop_pair INT TERM

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
    --device 0 --shard-index 0 --shard-count 2 --run-suffix spectrum-shard-000 \
    "${LIMIT_ARGS[@]}" "$@" &
  PID0=$!
  timeout 8h python -m jclosure.experiments.geometry_v3 \
    --config "$CONFIG" --stage spectrum --bank-manifest "$BANK_MANIFEST" \
    --device 1 --shard-index 1 --shard-count 2 --run-suffix spectrum-shard-001 \
    "${LIMIT_ARGS[@]}" "$@" &
  PID1=$!
  wait_pair
fi

if [[ "$STAGE" == "pareto" || "$STAGE" == "all" ]]; then
  timeout 12h python -m jclosure.experiments.geometry_v3 \
    --config "$CONFIG" --stage pareto --bank-manifest "$BANK_MANIFEST" \
    --device 0 --shard-index 0 --shard-count 2 --run-suffix pareto-shard-000 \
    "${LIMIT_ARGS[@]}" "$@" &
  PID0=$!
  timeout 12h python -m jclosure.experiments.geometry_v3 \
    --config "$CONFIG" --stage pareto --bank-manifest "$BANK_MANIFEST" \
    --device 1 --shard-index 1 --shard-count 2 --run-suffix pareto-shard-001 \
    "${LIMIT_ARGS[@]}" "$@" &
  PID1=$!
  wait_pair
fi
