#!/usr/bin/env bash
set -euo pipefail

export JCLOSURE_MODEL_DIR="${JCLOSURE_MODEL_DIR:-/data/CSK/J-space-project/models/Qwen3.5-4B-851bf6e}"
export JCLOSURE_ARTIFACT_DIR="${JCLOSURE_ARTIFACT_DIR:-/data/CSK/J-space-project/.jclosure-artifacts}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

stage="${1:?usage: scripts/run_persistent_channels_v7.sh guard|freeze|schema|restore|attribution|analyze}"
shift
python -m jclosure.experiments.persistent_channels_v7 \
  --config configs/persistent_channels_v7.yaml --stage "$stage" "$@"
