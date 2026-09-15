#!/usr/bin/env bash
set -euo pipefail

export JCLOSURE_MODEL_DIR="${JCLOSURE_MODEL_DIR:-/data/CSK/J-space-project/models/Qwen3.5-4B-851bf6e}"
export JCLOSURE_ARTIFACT_DIR="${JCLOSURE_ARTIFACT_DIR:-/data/CSK/J-space-project/.jclosure-artifacts}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

stage="${1:?usage: scripts/run_persistent_state_v8.sh generate|calibrate|formal|guard|teacher|freeze|bank|capture|analyze|features|compress|report}"
shift
case "$stage" in
  generate|calibrate|formal)
    module="jclosure.experiments.calibrate_v8"
    ;;
  guard|teacher|teacher-reparse|freeze|bank|capture|analyze)
    module="jclosure.experiments.persistent_state_v8"
    ;;
  features)
    module="jclosure.experiments.compress_persistent_v8"
    stage="features"
    ;;
  compress)
    module="jclosure.experiments.compress_persistent_v8"
    stage="analyze"
    ;;
  report)
    exec python -m jclosure.experiments.report_v8 \
      --config configs/persistent_state_v8.yaml "$@"
    ;;
  *)
    echo "unknown v8 stage: $stage" >&2
    exit 2
    ;;
esac
python -m "$module" --config configs/persistent_state_v8.yaml --stage "$stage" "$@"
