# J-state geometry audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: run + validate
- Date: 2026-08-29
- Verification Status: ANALYZED
- Protocol: exploratory protocol v3
- Baseline: d504eaa14af45f9df32101cf4599c55d3fac8707

## Status

This report is generated from saved Parquet records. Phase 0 v2 and its 0/1400
strict clamp result were not modified or re-adjudicated.

- Execution scope: formal
- Successful smoke run IDs: ['geometry-v3-20260829T133605Z-1bf9a00a-s20260828']

Dense state-definition feasibility warning: the local dense profile is near-injective or no natural strict candidate reached the frozen 0.20 displacement. This is not H1 evidence and triggers low-dimensional search.

## Map spectra

- M=4096, layer 23, A: rank@1e-4=2555/2560, stable rank=7.494, status=NUMERICALLY_UNRESOLVED
- M=4096, layer 23, CA: rank@1e-4=2556/2560, stable rank=24.596, status=NUMERICALLY_UNRESOLVED
- M=4096, layer 24, A: rank@1e-4=2555/2560, stable rank=8.583, status=NUMERICALLY_UNRESOLVED
- M=4096, layer 24, CA: rank@1e-4=2557/2560, stable rank=31.346, status=NUMERICALLY_UNRESOLVED
- M=4096, layer 25, A: rank@1e-4=2555/2560, stable rank=9.349, status=NUMERICALLY_UNRESOLVED
- M=4096, layer 25, CA: rank@1e-4=2558/2560, stable rank=34.322, status=NUMERICALLY_UNRESOLVED
- M=4096, layer 26, A: rank@1e-4=2555/2560, stable rank=10.161, status=NUMERICALLY_UNRESOLVED
- M=4096, layer 26, CA: rank@1e-4=2557/2560, stable rank=38.329, status=NUMERICALLY_UNRESOLVED
- M=4096, layer 27, A: rank@1e-4=2557/2560, stable rank=10.489, status=NUMERICALLY_UNRESOLVED
- M=4096, layer 27, CA: rank@1e-4=2558/2560, stable rank=40.386, status=NUMERICALLY_UNRESOLVED
- M=4096, layer 28, A: rank@1e-4=2556/2560, stable rank=9.914, status=NUMERICALLY_UNRESOLVED
- M=4096, layer 28, CA: rank@1e-4=2558/2560, stable rank=40.293, status=NUMERICALLY_UNRESOLVED
- M=4096, layer 29, A: rank@1e-4=2557/2560, stable rank=8.890, status=NUMERICALLY_UNRESOLVED
- M=4096, layer 29, CA: rank@1e-4=2558/2560, stable rank=33.628, status=NUMERICALLY_UNRESOLVED
- M=8192, layer 23, A: rank@1e-4=2556/2560, stable rank=8.208, status=NUMERICALLY_UNRESOLVED
- M=8192, layer 23, CA: rank@1e-4=2558/2560, stable rank=27.837, status=NUMERICALLY_UNRESOLVED
- M=8192, layer 24, A: rank@1e-4=2557/2560, stable rank=9.350, status=NUMERICALLY_UNRESOLVED
- M=8192, layer 24, CA: rank@1e-4=2558/2560, stable rank=35.563, status=NUMERICALLY_UNRESOLVED
- M=8192, layer 25, A: rank@1e-4=2557/2560, stable rank=10.147, status=NUMERICALLY_UNRESOLVED
- M=8192, layer 25, CA: rank@1e-4=2558/2560, stable rank=38.997, status=NUMERICALLY_UNRESOLVED
- M=8192, layer 26, A: rank@1e-4=2557/2560, stable rank=10.958, status=NUMERICALLY_UNRESOLVED
- M=8192, layer 26, CA: rank@1e-4=2558/2560, stable rank=43.880, status=NUMERICALLY_UNRESOLVED
- M=8192, layer 27, A: rank@1e-4=2557/2560, stable rank=11.226, status=NUMERICALLY_UNRESOLVED
- M=8192, layer 27, CA: rank@1e-4=2558/2560, stable rank=46.175, status=NUMERICALLY_UNRESOLVED
- M=8192, layer 28, A: rank@1e-4=2558/2560, stable rank=10.500, status=NUMERICALLY_UNRESOLVED
- M=8192, layer 28, CA: rank@1e-4=2559/2560, stable rank=45.662, status=NUMERICALLY_UNRESOLVED
- M=8192, layer 29, A: rank@1e-4=2558/2560, stable rank=9.329, status=NUMERICALLY_UNRESOLVED
- M=8192, layer 29, CA: rank@1e-4=2558/2560, stable rank=38.037, status=NUMERICALLY_UNRESOLVED
- M=16384, layer 23, A: rank@1e-4=2556/2560, stable rank=8.862, status=NUMERICALLY_UNRESOLVED
- M=16384, layer 23, CA: rank@1e-4=2558/2560, stable rank=32.705, status=NUMERICALLY_UNRESOLVED
- M=16384, layer 24, A: rank@1e-4=2557/2560, stable rank=10.035, status=NUMERICALLY_UNRESOLVED
- M=16384, layer 24, CA: rank@1e-4=2558/2560, stable rank=41.709, status=NUMERICALLY_UNRESOLVED
- M=16384, layer 25, A: rank@1e-4=2557/2560, stable rank=10.829, status=NUMERICALLY_UNRESOLVED
- M=16384, layer 25, CA: rank@1e-4=2558/2560, stable rank=46.010, status=NUMERICALLY_UNRESOLVED
- M=16384, layer 26, A: rank@1e-4=2557/2560, stable rank=11.583, status=NUMERICALLY_UNRESOLVED
- M=16384, layer 26, CA: rank@1e-4=2558/2560, stable rank=52.079, status=NUMERICALLY_UNRESOLVED
- M=16384, layer 27, A: rank@1e-4=2558/2560, stable rank=11.744, status=NUMERICALLY_UNRESOLVED
- M=16384, layer 27, CA: rank@1e-4=2559/2560, stable rank=55.283, status=NUMERICALLY_UNRESOLVED
- M=16384, layer 28, A: rank@1e-4=2558/2560, stable rank=10.872, status=NUMERICALLY_UNRESOLVED
- M=16384, layer 28, CA: rank@1e-4=2559/2560, stable rank=54.437, status=NUMERICALLY_UNRESOLVED
- M=16384, layer 29, A: rank@1e-4=2558/2560, stable rank=9.532, status=NUMERICALLY_UNRESOLVED
- M=16384, layer 29, CA: rank@1e-4=2559/2560, stable rank=44.935, status=NUMERICALLY_UNRESOLVED

## Local normalized-state checks

- Local rows: 5376
- Analytic/autograd JVP/VJP checked rows: 5376
- Rows failing the frozen 1e-4 relative-error check: 0
- Maximum normalized radial residual: 4.84272e-08

## Pareto audit

- Candidate rows: 0
- Source files: 0 Pareto and 4 spectrum Parquet files
- Failed run manifests retained: 5

All thresholds are protocol constants. No behavioral H1/H2/H3 conclusion is drawn
from geometry or construction feasibility alone.
