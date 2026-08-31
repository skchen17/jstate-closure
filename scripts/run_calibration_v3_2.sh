#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/closure_v3_2.yaml}"
STAGE="${STAGE:-all}"
GROUP="${JCLOSURE_SHARD_GROUP_ID:-calibration-v3-2-$(date -u +%Y%m%dT%H%M%SZ)}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then LIMIT_ARGS=(--limit "$LIMIT"); fi
if [[ -z "${JCLOSURE_GIT_WORKTREE_SNAPSHOT:-}" ]]; then
  JCLOSURE_GIT_WORKTREE_SNAPSHOT="$(python -c 'import json; from jclosure.provenance import git_worktree_snapshot; print(json.dumps(git_worktree_snapshot("."), sort_keys=True))')"
  export JCLOSURE_GIT_WORKTREE_SNAPSHOT
fi
if [[ "$STAGE" == "bank" || "$STAGE" == "all" ]]; then
  timeout 12h python -m jclosure.experiments.calibrate_v3_2 --config "$CONFIG" --stage bank "${LIMIT_ARGS[@]}"
fi
BANK_MANIFEST="${BANK_MANIFEST:-$(find results/v3_2/raw -path '*/activation_bank_manifest.jsonl' -type f | sort | tail -n 1)}"
if [[ -z "$BANK_MANIFEST" ]]; then echo "v3.2 activation bank missing" >&2; exit 2; fi
if [[ "$STAGE" == "bank" ]]; then exit 0; fi
export JCLOSURE_SHARD_GROUP_ID="$GROUP"
timeout 12h python -m jclosure.experiments.calibrate_v3_2 --config "$CONFIG" --stage calibrate --bank-manifest "$BANK_MANIFEST" --device 0 --shard-index 0 --shard-count 2 --shard-group-id "$GROUP" --run-suffix shard-000 "${LIMIT_ARGS[@]}" & PID0=$!
timeout 12h python -m jclosure.experiments.calibrate_v3_2 --config "$CONFIG" --stage calibrate --bank-manifest "$BANK_MANIFEST" --device 1 --shard-index 1 --shard-count 2 --shard-group-id "$GROUP" --run-suffix shard-001 "${LIMIT_ARGS[@]}" & PID1=$!
trap 'kill -INT "$PID0" "$PID1" 2>/dev/null || true' INT TERM
set +e; wait "$PID0"; S0=$?; wait "$PID1"; S1=$?; set -e
if (( S0 != 0 || S1 != 0 )); then echo "v3.2 calibration failed: $S0/$S1" >&2; exit 1; fi
timeout 12h python -m jclosure.experiments.calibrate_v3_2 --config "$CONFIG" --stage merge --bank-manifest "$BANK_MANIFEST" --shard-count 2 --shard-group-id "$GROUP"
