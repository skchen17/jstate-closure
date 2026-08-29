#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/closure_v3_confirm.yaml}"
export CONFIG
exec scripts/run_closure_v3_confirm.sh "$@"
