#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
PYTHON_BIN="${PYTHON_BIN:-/home/user/anaconda3/bin/python}"
FEATURE_GPU="${V13_FEATURE_GPU:-1}"
JVP_GPUS="${V13_JVP_GPUS:-0,1}"
export HF_HOME="${HF_HOME:-/data/CSK/J-space-project/.hf-cache}"

MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES="$FEATURE_GPU" \
  "$PYTHON_BIN" -m jclosure.experiments.runtime_v13_features_v15 --target freeze
MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES="$FEATURE_GPU" \
  "$PYTHON_BIN" -m jclosure.experiments.runtime_v13_features_v15 \
  --target geometry --stage features --run-suffix features-memory-amendment-15
"$PYTHON_BIN" -m jclosure.experiments.geometry_v13 \
  --stage scaling --run-suffix scaling
MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES="$FEATURE_GPU" \
  "$PYTHON_BIN" -m jclosure.experiments.runtime_v13_features_v15 \
  --target geometry --stage directions --run-suffix directions
"$PYTHON_BIN" -m jclosure.experiments.geometry_v13 \
  --stage freeze-jvp --run-suffix freeze-jvp
"$PYTHON_BIN" -m jclosure.experiments.runtime_v13 --target freeze
"$PYTHON_BIN" -m jclosure.experiments.runtime_v13_jvp_scalar_sharded \
  --freeze-amendment
MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES="$JVP_GPUS" \
  "$PYTHON_BIN" -m jclosure.experiments.runtime_v13_jvp_scalar_sharded \
  --stage jvp --run-suffix exact-scalar-shard-0 --shard-index 0 &
JVP_SHARD_0_PID=$!
MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES="$JVP_GPUS" \
  "$PYTHON_BIN" -m jclosure.experiments.runtime_v13_jvp_scalar_sharded \
  --stage jvp --run-suffix exact-scalar-shard-1 --shard-index 1 &
JVP_SHARD_1_PID=$!
wait "$JVP_SHARD_0_PID"
wait "$JVP_SHARD_1_PID"
"$PYTHON_BIN" scripts/merge_v13_jvp_scalar_shards.py
"$PYTHON_BIN" -m jclosure.experiments.runtime_v13 \
  --target geometry --stage oracle-development --run-suffix oracle-development
"$PYTHON_BIN" -m jclosure.experiments.geometry_v13 \
  --stage freeze-finalists --run-suffix freeze-finalists
"$PYTHON_BIN" -m jclosure.experiments.runtime_v13 \
  --target geometry --stage linearity --run-suffix linearity
"$PYTHON_BIN" -m jclosure.experiments.runtime_v13 \
  --target geometry --stage oracle-confirmatory --run-suffix oracle-confirmatory
"$PYTHON_BIN" -m jclosure.experiments.geometry_v13 \
  --stage analyze --run-suffix analyze
"$PYTHON_BIN" -m jclosure.reporting_v13
"$PYTHON_BIN" scripts/build_complete_version_report.py V13
