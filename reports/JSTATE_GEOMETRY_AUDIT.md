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
- Successful smoke run IDs: ['geometry-v3-20260829T133605Z-1bf9a00a-s20260828', 'geometry-v3-20260829T155433Z-3b3fa742-s20260828', 'geometry-v3-20260829T162846Z-3b3fa742-s20260828']

Dense state-definition feasibility warning: the median local rank at 1e-4 is 2557/2560 and the median tangent-null dimension is 2. The normalized dense profile is therefore operationally near-injective under the frozen rule. This is not compact H1 evidence and triggers low-dimensional search.

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

- Candidate rows: 45360
- Source files: 2 Pareto and 4 spectrum Parquet files
- Failed run manifests retained: 15
- Canonical state-definition-aware summary: results/v3/processed/pareto_formal_summary_v3.parquet

The dense formal rows below use the frozen `1e-4 × sigma_max` tolerance. Sparse
rows use their independent support/coefficient/reconstruction equality gate;
dense cosine is only a sensitivity metric for V3-Sparse.

### Formal construction attrition

- M=4096, hard_constrained: constructed 504/1008, state-equal 504, natural+equal 504, formal rows 224
- M=4096, norm_tangent_dense_null: constructed 522/1008, state-equal 522, natural+equal 504, formal rows 224
- M=4096, sparse_remainder: constructed 1008/1008, state-equal 608, natural+equal 396, formal rows 71
- M=8192, hard_constrained: constructed 369/1008, state-equal 369, natural+equal 369, formal rows 164
- M=8192, norm_tangent_dense_null: constructed 387/1008, state-equal 387, natural+equal 369, formal rows 164
- M=8192, sparse_remainder: constructed 1008/1008, state-equal 542, natural+equal 384, formal rows 63
- M=16384, hard_constrained: constructed 549/1008, state-equal 549, natural+equal 549, formal rows 244
- M=16384, norm_tangent_dense_null: constructed 567/1008, state-equal 567, natural+equal 549, formal rows 244
- M=16384, sparse_remainder: constructed 1008/1008, state-equal 471, natural+equal 355, formal rows 61

### Layer-by-layer formal feasibility

`formal anchors` counts anchors with at least one natural, state-equal candidate
at displacement `>=0.20`; `max displacement` is computed before imposing that
minimum so infeasible cells remain visible.

| M | Layer | Method | constructed/attempted rows | state-equal rows | natural+equal rows | formal anchors | max displacement |
|---:|---:|:---|---:|---:|---:|---:|---:|
| 4096 | 23 | Dense optimized | 117/144 | 117 | 117 | 13/16 | 0.500001 |
| 4096 | 23 | Dense local-null | 135/144 | 135 | 117 | 13/16 | 0.500001 |
| 4096 | 23 | Sparse same-definition | 144/144 | 91 | 58 | 5/16 | 0.521739 |
| 4096 | 24 | Dense optimized | 63/144 | 63 | 63 | 7/16 | 0.500001 |
| 4096 | 24 | Dense local-null | 63/144 | 63 | 63 | 7/16 | 0.500001 |
| 4096 | 24 | Sparse same-definition | 144/144 | 82 | 51 | 5/16 | 0.503715 |
| 4096 | 25 | Dense optimized | 81/144 | 81 | 81 | 9/16 | 0.500001 |
| 4096 | 25 | Dense local-null | 81/144 | 81 | 81 | 9/16 | 0.500001 |
| 4096 | 25 | Sparse same-definition | 144/144 | 71 | 42 | 2/16 | 0.494247 |
| 4096 | 26 | Dense optimized | 36/144 | 36 | 36 | 4/16 | 0.500001 |
| 4096 | 26 | Dense local-null | 36/144 | 36 | 36 | 4/16 | 0.500001 |
| 4096 | 26 | Sparse same-definition | 144/144 | 89 | 59 | 2/16 | 0.498702 |
| 4096 | 27 | Dense optimized | 45/144 | 45 | 45 | 5/16 | 0.500001 |
| 4096 | 27 | Dense local-null | 45/144 | 45 | 45 | 5/16 | 0.500001 |
| 4096 | 27 | Sparse same-definition | 144/144 | 91 | 61 | 4/16 | 0.492990 |
| 4096 | 28 | Dense optimized | 45/144 | 45 | 45 | 5/16 | 0.500001 |
| 4096 | 28 | Dense local-null | 45/144 | 45 | 45 | 5/16 | 0.500001 |
| 4096 | 28 | Sparse same-definition | 144/144 | 89 | 59 | 5/16 | 0.484350 |
| 4096 | 29 | Dense optimized | 117/144 | 117 | 117 | 13/16 | 0.500001 |
| 4096 | 29 | Dense local-null | 117/144 | 117 | 117 | 13/16 | 0.500001 |
| 4096 | 29 | Sparse same-definition | 144/144 | 95 | 66 | 4/16 | 0.485842 |
| 8192 | 23 | Dense optimized | 0/144 | 0 | 0 | 0/16 | none |
| 8192 | 23 | Dense local-null | 9/144 | 9 | 0 | 0/16 | none |
| 8192 | 23 | Sparse same-definition | 144/144 | 74 | 45 | 4/16 | 0.492353 |
| 8192 | 24 | Dense optimized | 18/144 | 18 | 18 | 2/16 | 0.500001 |
| 8192 | 24 | Dense local-null | 27/144 | 27 | 18 | 2/16 | 0.500001 |
| 8192 | 24 | Sparse same-definition | 144/144 | 75 | 49 | 4/16 | 0.296329 |
| 8192 | 25 | Dense optimized | 27/144 | 27 | 27 | 3/16 | 0.500001 |
| 8192 | 25 | Dense local-null | 27/144 | 27 | 27 | 3/16 | 0.500001 |
| 8192 | 25 | Sparse same-definition | 144/144 | 64 | 44 | 3/16 | 0.522463 |
| 8192 | 26 | Dense optimized | 36/144 | 36 | 36 | 4/16 | 0.500001 |
| 8192 | 26 | Dense local-null | 36/144 | 36 | 36 | 4/16 | 0.500001 |
| 8192 | 26 | Sparse same-definition | 144/144 | 82 | 62 | 3/16 | 0.491986 |
| 8192 | 27 | Dense optimized | 108/144 | 108 | 108 | 12/16 | 0.500001 |
| 8192 | 27 | Dense local-null | 108/144 | 108 | 108 | 12/16 | 0.500001 |
| 8192 | 27 | Sparse same-definition | 144/144 | 66 | 48 | 3/16 | 0.490987 |
| 8192 | 28 | Dense optimized | 45/144 | 45 | 45 | 5/16 | 0.500001 |
| 8192 | 28 | Dense local-null | 45/144 | 45 | 45 | 5/16 | 0.500000 |
| 8192 | 28 | Sparse same-definition | 144/144 | 89 | 70 | 7/16 | 0.518333 |
| 8192 | 29 | Dense optimized | 135/144 | 135 | 135 | 15/16 | 0.500001 |
| 8192 | 29 | Dense local-null | 135/144 | 135 | 135 | 15/16 | 0.500001 |
| 8192 | 29 | Sparse same-definition | 144/144 | 92 | 66 | 4/16 | 0.465578 |
| 16384 | 23 | Dense optimized | 63/144 | 63 | 63 | 7/16 | 0.500001 |
| 16384 | 23 | Dense local-null | 72/144 | 72 | 63 | 7/16 | 0.500001 |
| 16384 | 23 | Sparse same-definition | 144/144 | 63 | 45 | 5/16 | 0.298333 |
| 16384 | 24 | Dense optimized | 81/144 | 81 | 81 | 9/16 | 0.500001 |
| 16384 | 24 | Dense local-null | 90/144 | 90 | 81 | 9/16 | 0.500001 |
| 16384 | 24 | Sparse same-definition | 144/144 | 66 | 49 | 5/16 | 0.459484 |
| 16384 | 25 | Dense optimized | 90/144 | 90 | 90 | 10/16 | 0.500000 |
| 16384 | 25 | Dense local-null | 90/144 | 90 | 90 | 10/16 | 0.500000 |
| 16384 | 25 | Sparse same-definition | 144/144 | 69 | 52 | 4/16 | 0.521863 |
| 16384 | 26 | Dense optimized | 90/144 | 90 | 90 | 10/16 | 0.500001 |
| 16384 | 26 | Dense local-null | 90/144 | 90 | 90 | 10/16 | 0.500001 |
| 16384 | 26 | Sparse same-definition | 144/144 | 53 | 43 | 5/16 | 0.323262 |
| 16384 | 27 | Dense optimized | 81/144 | 81 | 81 | 9/16 | 0.500001 |
| 16384 | 27 | Dense local-null | 81/144 | 81 | 81 | 9/16 | 0.500001 |
| 16384 | 27 | Sparse same-definition | 144/144 | 61 | 52 | 4/16 | 0.518020 |
| 16384 | 28 | Dense optimized | 0/144 | 0 | 0 | 0/16 | none |
| 16384 | 28 | Dense local-null | 0/144 | 0 | 0 | 0/16 | none |
| 16384 | 28 | Sparse same-definition | 144/144 | 67 | 53 | 3/16 | 0.494150 |
| 16384 | 29 | Dense optimized | 144/144 | 144 | 144 | 16/16 | 0.500001 |
| 16384 | 29 | Dense local-null | 144/144 | 144 | 144 | 16/16 | 0.500001 |
| 16384 | 29 | Sparse same-definition | 144/144 | 92 | 61 | 2/16 | 0.471097 |

All thresholds are protocol constants. No behavioral H1/H2/H3 conclusion is drawn
from geometry or construction feasibility alone.
