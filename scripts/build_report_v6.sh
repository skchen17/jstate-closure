#!/usr/bin/env bash
set -euo pipefail
python -m jclosure.experiments.report_v6 --config configs/peripheral_v6.yaml "$@"
