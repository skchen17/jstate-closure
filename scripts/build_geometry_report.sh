#!/usr/bin/env bash
set -euo pipefail
python -m jclosure.reporting_v3 --config "${CONFIG:-configs/geometry_v3.yaml}" "$@"
