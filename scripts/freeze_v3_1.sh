#!/usr/bin/env bash
set -euo pipefail
KIND="${1:-memory}"
shift || true
python -m jclosure.experiments.freeze_v3_1 --kind "$KIND" "$@"
