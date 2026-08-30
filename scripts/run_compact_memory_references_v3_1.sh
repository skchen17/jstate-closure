#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/compact_memory_v3_1.yaml}

for DIMENSION in 64 128 256 512; do
  for SEED in 20260828 20260829 20260830; do
    timeout 72h python -m jclosure.experiments.compact_memory_references_v3_1 \
      --config "$CONFIG" \
      --stage references \
      --dimension "$DIMENSION" \
      --controller-seed "$SEED"
  done
done

python -m jclosure.experiments.compact_memory_references_v3_1 \
  --config "$CONFIG" \
  --stage fidelity
