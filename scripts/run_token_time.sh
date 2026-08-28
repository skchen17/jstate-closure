#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/confirm_v2.yaml}"
python -m jclosure.experiments.token_time_closure --config "$CONFIG" ${ARGS:-}
