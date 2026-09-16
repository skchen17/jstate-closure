#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/user/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="${PYTHONPATH:-}:src"

python -m jclosure.experiments.finalize_reports_v10 "$@"
