#!/usr/bin/env bash
set -euo pipefail
python -m jclosure.experiments.prepare_v3_2
python -m jclosure.experiments.freeze_v3_2 --kind calibration
python -m jclosure.experiments.freeze_v3_2 --kind memory
