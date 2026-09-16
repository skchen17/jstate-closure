#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/user/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${PWD}/src"
export HF_HOME="/data/CSK/J-space-project/.hf-cache"

stage="${1:?usage: $0 STAGE [extra args]}"
shift

case "${stage}" in
  freeze-base|fit-models|prepare-development|prepare-scaling|freeze-prepared|prepare-confirmatory)
    module="jclosure.experiments.prepare_v12"
    ;;
  measurement-audit|scaling-causal|oracle-stage1|freeze-stage2|oracle-stage2|freeze-confirm|confirmatory|strict-replacement)
    module="jclosure.experiments.causal_v12"
    ;;
  prepare-directions|freeze-jvp|run-jvp)
    module="jclosure.experiments.jvp_v12"
    ;;
  analyze)
    exec /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v12 "$@"
    ;;
  report)
    exec /home/user/anaconda3/bin/python -m jclosure.reporting_v12 "$@"
    ;;
  test)
    exec /home/user/anaconda3/bin/python -m pytest -q tests/test_v12.py "$@"
    ;;
  *)
    echo "unknown v12 stage: ${stage}" >&2
    exit 2
    ;;
esac

# The local cuda:0 below maps to physical GPU 1 by default. Override explicitly if needed.
export CUDA_VISIBLE_DEVICES="${V12_CUDA_VISIBLE_DEVICES:-1}"
exec /home/user/anaconda3/bin/python -m "${module}" --stage "${stage}" "$@"
