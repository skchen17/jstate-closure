#!/usr/bin/env bash
set -euo pipefail

python -m jclosure.experiments.report_v7 \
  --config configs/persistent_channels_v7_corrective.yaml "$@"
