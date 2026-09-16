#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/user/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${PYTHONPATH:-}:src"

stage="${1:?stage required: freeze|audit}"
shift

python -m jclosure.experiments.strict_interface_v10 --stage "${stage}" "$@"
