#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: run_sufficiency_v9.sh <freeze|analyze|report> [extra args]}"
shift
case "${stage}" in
  freeze|analyze)
    exec python -m jclosure.experiments.sufficiency_v9 \
      --config configs/sufficiency_v9.yaml \
      --stage "${stage}" \
      "$@"
    ;;
  report)
    exec python -m jclosure.experiments.report_v9 \
      --config configs/sufficiency_v9.yaml \
      "$@"
    ;;
  *)
    echo "unknown v9 stage: ${stage}" >&2
    exit 2
    ;;
esac
