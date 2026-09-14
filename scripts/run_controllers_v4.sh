#!/usr/bin/env bash
set -euo pipefail

config="${CONFIG:-configs/predictive_v4.yaml}"
for seed in 20260828 20260829 20260830; do
  python -m jclosure.experiments.controllers_v4 --config "$config" --stage train --family markov --history 1 --memory-dim 0 --controller-seed "$seed" --run-suffix "markov-s${seed}"
  for history in 1 2 4 8 16; do
    python -m jclosure.experiments.controllers_v4 --config "$config" --stage train --family history --history "$history" --memory-dim 0 --controller-seed "$seed" --run-suffix "history-h${history}-s${seed}"
  done
  for memory in 16 32 64 128 256; do
    python -m jclosure.experiments.controllers_v4 --config "$config" --stage train --family gru --history 1 --memory-dim "$memory" --controller-seed "$seed" --run-suffix "gru-m${memory}-s${seed}"
  done
done
python -m jclosure.experiments.controllers_v4 --config "$config" --stage aggregate --run-suffix aggregate
