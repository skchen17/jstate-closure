#!/usr/bin/env bash
set -euo pipefail

stage="${1:?usage: run_causal_sufficiency_v10.sh <freeze|audit|freeze-candidates|prepare-decoder|causal|report> [extra args]}"
shift
case "${stage}" in
  freeze|audit|freeze-candidates)
    exec python -m jclosure.experiments.sufficiency_v10 \
      --config configs/causal_sufficiency_v10.yaml \
      --stage "${stage}" \
      "$@"
    ;;
  prepare-decoder|causal)
    exec python -m jclosure.experiments.decoded_causal_v10 \
      --config configs/causal_sufficiency_v10.yaml \
      --stage "${stage}" \
      "$@"
    ;;
  report)
    exec python -m jclosure.experiments.report_v10 \
      --config configs/causal_sufficiency_v10.yaml \
      "$@"
    ;;
  *)
    echo "unknown v10 stage: ${stage}" >&2
    exit 2
    ;;
esac
