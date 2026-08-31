#!/usr/bin/env bash
set -euo pipefail
python -m jclosure.reporting_v3_2
python -m jclosure.reporting_postrun_v3_2
