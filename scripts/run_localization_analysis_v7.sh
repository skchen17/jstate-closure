#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: scripts/run_localization_analysis_v7.sh freeze|analyze}"
shift
python -m jclosure.experiments.localization_analysis_v7 \
  --config configs/persistent_channels_v7.yaml --stage "$stage" "$@"
