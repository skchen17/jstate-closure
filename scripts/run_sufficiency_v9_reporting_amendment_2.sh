#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: run_sufficiency_v9_reporting_amendment_2.sh <freeze|report> [extra args]}"
shift
exec python -m jclosure.experiments.report_v9_amendment_2 \
  --config configs/sufficiency_v9.yaml \
  --stage "${stage}" \
  "$@"
