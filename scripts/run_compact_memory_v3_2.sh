#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/compact_memory_v3_2.yaml}"
STAGE="${STAGE:-all}"
if [[ "$STAGE" == "merge-audit" || "$STAGE" == "all" ]]; then
  timeout 72h python -m jclosure.experiments.compact_memory_v3_2 --config "$CONFIG" --stage merge-audit
fi
if [[ "$STAGE" == "merge-audit" ]]; then exit 0; fi
if [[ "$STAGE" == "screen" || "$STAGE" == "all" ]]; then
  timeout 72h python -m jclosure.experiments.compact_memory_v3_2 --config "$CONFIG" --stage screen --device 0
fi
if [[ "$STAGE" == "screen" ]]; then exit 0; fi
if ! jq -e '.temporal_training_authorized == true' results/v3_2/processed/compact_memory_representation_screen_v3_2.json >/dev/null; then
  echo "Temporal training gated by v3.2 representation screen" >&2; exit 0
fi
for SEED in 20260828 20260829 20260830; do
  (
    timeout 72h python -m jclosure.experiments.compact_memory_v3_2 --config "$CONFIG" --stage train --device 0 --family markov --controller-seed "$SEED"
    for HISTORY in 1 2 4 8 16; do
      timeout 72h python -m jclosure.experiments.compact_memory_v3_2 --config "$CONFIG" --stage train --device 0 --family history --history "$HISTORY" --controller-seed "$SEED"
    done
  ) & PID0=$!
  (
    for MEMORY in 16 32 64 128 256; do
      timeout 72h python -m jclosure.experiments.compact_memory_v3_2 --config "$CONFIG" --stage train --device 1 --family gru --memory-dimension "$MEMORY" --controller-seed "$SEED"
    done
  ) & PID1=$!
  set +e; wait "$PID0"; S0=$?; wait "$PID1"; S1=$?; set -e
  if (( S0 != 0 || S1 != 0 )); then echo "compact-memory v3.2 training failed: $S0/$S1" >&2; exit 1; fi
done
timeout 72h python -m jclosure.experiments.compact_memory_v3_2 --config "$CONFIG" --stage train --device 0 --family gru --memory-dimension 128 --controller-seed 20260828 --training-subset teacher_correct_only
