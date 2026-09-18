#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
PYTHON_BIN="${PYTHON_BIN:-/home/user/anaconda3/bin/python}"

run_bank() {
  "$PYTHON_BIN" -m jclosure.experiments.bank_v13 "$@"
}

run_geometry() {
  "$PYTHON_BIN" -m jclosure.experiments.geometry_v13 "$@"
}

case "${1:-all}" in
  prepare-bank)
    run_bank --stage generate --run-suffix generate
    run_bank --stage teacher --run-suffix teacher
    run_bank --stage select --run-suffix select
    run_bank --stage guard --run-suffix guard
    run_bank --stage freeze --run-suffix freeze
    ;;
  capture)
    run_bank --stage capture --split train --run-suffix capture-train
    run_bank --stage capture --split validation --run-suffix capture-validation
    run_bank --stage capture --split final_test --run-suffix capture-final
    run_bank --stage freeze-capture --run-suffix freeze-capture
    ;;
  geometry)
    run_geometry --stage features --run-suffix features
    run_geometry --stage scaling --run-suffix scaling
    run_geometry --stage directions --run-suffix directions
    run_geometry --stage freeze-jvp --run-suffix freeze-jvp
    "$PYTHON_BIN" -m jclosure.experiments.runtime_v13 --target freeze
    "$PYTHON_BIN" -m jclosure.experiments.runtime_v13 --target geometry --stage jvp --run-suffix exact-jvp
    "$PYTHON_BIN" -m jclosure.experiments.runtime_v13 --target geometry --stage oracle-development --run-suffix oracle-development
    run_geometry --stage freeze-finalists --run-suffix freeze-finalists
    "$PYTHON_BIN" -m jclosure.experiments.runtime_v13 --target geometry --stage linearity --run-suffix linearity
    "$PYTHON_BIN" -m jclosure.experiments.runtime_v13 --target geometry --stage oracle-confirmatory --run-suffix oracle-confirmatory
    run_geometry --stage analyze --run-suffix analyze
    "$PYTHON_BIN" -m jclosure.reporting_v13
    "$PYTHON_BIN" scripts/build_complete_version_report.py V13
    ;;
  *)
    echo "Use prepare-bank, capture, or geometry; stages are intentionally resumable." >&2
    exit 2
    ;;
esac
