#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/qwen3_5_4b.yaml}"
python -m jclosure.experiments.validate_lens --config "$CONFIG" ${ARGS:-}
