#!/usr/bin/env bash
set -euo pipefail

export JCLOSURE_MODEL_DIR="${JCLOSURE_MODEL_DIR:-/data/CSK/J-space-project/models/Qwen3.5-4B-851bf6e}"
export JCLOSURE_ARTIFACT_DIR="${JCLOSURE_ARTIFACT_DIR:-/data/CSK/J-space-project/.jclosure-artifacts}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

experiment="${EXPERIMENT:-single}"
if [[ "$experiment" == "mediation" ]]; then
  python -m jclosure.experiments.mediation_v4 --config configs/predictive_v4.yaml "$@"
else
  python -m jclosure.experiments.causal_single_v4 --config configs/predictive_v4.yaml "$@"
fi
