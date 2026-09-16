#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/user/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${PYTHONPATH:-}:src"

stage="${1:?stage required: freeze|causal}"
shift

python -m jclosure.experiments.decoded_causal_v10_amendment --stage "${stage}" "$@"
