#!/usr/bin/env bash
set -euo pipefail

export JCLOSURE_MODEL_DIR="${JCLOSURE_MODEL_DIR:-/data/CSK/J-space-project/models/Qwen3.5-4B-851bf6e}"
export JCLOSURE_ARTIFACT_DIR="${JCLOSURE_ARTIFACT_DIR:-/data/CSK/J-space-project/.jclosure-artifacts}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

stage="${1:-traces}"
shift || true
case "$stage" in
  traces)
    python -m jclosure.experiments.traces_v4 --config configs/predictive_v4.yaml "$@"
    ;;
  screen)
    python -m jclosure.experiments.predictive_state_v4 --config configs/predictive_v4.yaml "$@"
    ;;
  controller)
    python -m jclosure.experiments.controllers_v4 --config configs/predictive_v4.yaml "$@"
    ;;
  reference)
    python -m jclosure.experiments.references_v4 --config configs/predictive_v4.yaml "$@"
    ;;
  *)
    echo "unknown v4 predictive stage: $stage" >&2
    exit 2
    ;;
esac
