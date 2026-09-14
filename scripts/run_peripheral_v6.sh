#!/usr/bin/env bash
set -euo pipefail

export JCLOSURE_MODEL_DIR="${JCLOSURE_MODEL_DIR:-/data/CSK/J-space-project/models/Qwen3.5-4B-851bf6e}"
export JCLOSURE_ARTIFACT_DIR="${JCLOSURE_ARTIFACT_DIR:-/data/CSK/J-space-project/.jclosure-artifacts}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

stage="${1:?usage: scripts/run_peripheral_v6.sh freeze|causal|merge|ceiling}"
shift
case "$stage" in
  freeze) python -m jclosure.experiments.causal_endpoint_v6 --config configs/peripheral_v6.yaml --stage freeze "$@" ;;
  causal) python -m jclosure.experiments.causal_endpoint_v6 --config configs/peripheral_v6.yaml --stage run "$@" ;;
  merge) python -m jclosure.experiments.causal_endpoint_v6 --config configs/peripheral_v6.yaml --stage merge "$@" ;;
  ceiling) python -m jclosure.experiments.peripheral_ceiling_v6 --config configs/peripheral_v6.yaml --stage run "$@" ;;
  compact) python -m jclosure.experiments.peripheral_ceiling_v6 --config configs/peripheral_v6.yaml --stage compact "$@" ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
