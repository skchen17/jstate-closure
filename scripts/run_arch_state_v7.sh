#!/usr/bin/env bash
set -euo pipefail

export JCLOSURE_MODEL_DIR="${JCLOSURE_MODEL_DIR:-/data/CSK/J-space-project/models/Qwen3.5-4B-851bf6e}"
export JCLOSURE_ARTIFACT_DIR="${JCLOSURE_ARTIFACT_DIR:-/data/CSK/J-space-project/.jclosure-artifacts}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

stage="${1:?usage: scripts/run_arch_state_v7.sh localization-freeze|localize|localization-analyze|compression-freeze|compress|compression-analyze}"
shift
case "$stage" in
  localization-freeze) python -m jclosure.experiments.localize_channels_v7 --config configs/persistent_channels_v7.yaml --stage freeze "$@" ;;
  localize) python -m jclosure.experiments.localize_channels_v7 --config configs/persistent_channels_v7.yaml --stage run "$@" ;;
  localization-analyze) python -m jclosure.experiments.localize_channels_v7 --config configs/persistent_channels_v7.yaml --stage analyze "$@" ;;
  compression-freeze) python -m jclosure.experiments.arch_compression_v7 --config configs/persistent_channels_v7.yaml --stage freeze "$@" ;;
  compress) python -m jclosure.experiments.arch_compression_v7 --config configs/persistent_channels_v7.yaml --stage run "$@" ;;
  compression-analyze) python -m jclosure.experiments.arch_compression_v7 --config configs/persistent_channels_v7.yaml --stage analyze "$@" ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
