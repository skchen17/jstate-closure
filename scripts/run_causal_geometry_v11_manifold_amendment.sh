#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/user/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${PWD}/src"
export HF_HOME="/data/CSK/J-space-project/.hf-cache"

stage="${1:?usage: $0 freeze|analyze|report}"
shift
exec /home/user/anaconda3/bin/python -m jclosure.experiments.analyze_v11_amendment \
  --stage "${stage}" "$@"
