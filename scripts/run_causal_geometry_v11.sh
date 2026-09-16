#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/user/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${PWD}/src"
export HF_HOME="/data/CSK/J-space-project/.hf-cache"

stage="${1:?usage: $0 STAGE [extra args]}"
shift

case "${stage}" in
  freeze-base|select-models|prepare-development|freeze-prepared|prepare-confirmatory)
    module="jclosure.experiments.prepare_v11"
    ;;
  channel-audit|oracle-development|factorized-stage1|freeze-stage2|factorized-stage2|freeze-confirm|confirmatory)
    module="jclosure.experiments.causal_v11"
    ;;
  analyze|report)
    module="jclosure.experiments.analyze_v11"
    ;;
  *)
    echo "unknown v11 stage: ${stage}" >&2
    exit 2
    ;;
esac

exec /home/user/anaconda3/bin/python -m "${module}" --stage "${stage}" "$@"
