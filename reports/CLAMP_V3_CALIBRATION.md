# Clamp v3 calibration

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: run + validate
- Date: 2026-08-29
- Verification Status: ANALYZED
- Protocol: exploratory protocol v3

## Gate

Calibration saved 4527/12600 formal-valid candidate records. Authorized protocols: none. Gate reasons: {'naturality_valid_fraction': 11, 'strict_valid_count': 57}.

- Merged run: `clamp-v3-calibration-20260830T023149Z-3b3fa742-s20260828`
- Source shards: clamp-v3-calibration-20260829T232049Z-3b3fa742-s20260828-clamp-shard-000, clamp-v3-calibration-20260829T232049Z-efb45693-s20260828-clamp-shard-001
- Hook sanity: `{"cleanup_exact": true, "determinism_exact": true, "finite": true, "identity_exact": true, "zero_exact": true}`

Formal validity requires the state-definition-specific equality gate, RMS drift
at most 0.02, displacement at least 0.20 of the natural scale, and the frozen
naturality envelope. Candidates between 0.05 and 0.20 are sensitivity records
only and cannot support H1/H2/H3.

The behavioral gate additionally requires at least four eligible increasing
layers so a selected L1 has at least three later eligible layers. No tested
state definition and dictionary size met that condition.

## Layer-by-layer calibration

`strict-valid` is evaluated before naturality; `formal-natural` additionally
passes the frozen 99% naturality envelope. All denominators are 200 paired
anchor/donor trials.

| M | Layer | Method | strict-valid | formal-natural | natural fraction | eligible | reasons |
|---:|---:|:---|---:|---:|---:|:---:|:---|
| 4096 | 23 | dense_local_null | 188/200 | 175/200 | 0.931 | no | naturality_valid_fraction |
| 4096 | 23 | dense_optimized | 175/200 | 175/200 | 1.000 | yes | - |
| 4096 | 23 | sparse_same_definition | 19/200 | 19/200 | 1.000 | no | strict_valid_count |
| 4096 | 24 | dense_local_null | 123/200 | 115/200 | 0.935 | no | strict_valid_count, naturality_valid_fraction |
| 4096 | 24 | dense_optimized | 115/200 | 115/200 | 1.000 | no | strict_valid_count |
| 4096 | 24 | sparse_same_definition | 23/200 | 23/200 | 1.000 | no | strict_valid_count |
| 4096 | 25 | dense_local_null | 78/200 | 72/200 | 0.923 | no | strict_valid_count, naturality_valid_fraction |
| 4096 | 25 | dense_optimized | 72/200 | 72/200 | 1.000 | no | strict_valid_count |
| 4096 | 25 | sparse_same_definition | 26/200 | 26/200 | 1.000 | no | strict_valid_count |
| 4096 | 26 | dense_local_null | 60/200 | 59/200 | 0.983 | no | strict_valid_count |
| 4096 | 26 | dense_optimized | 59/200 | 59/200 | 1.000 | no | strict_valid_count |
| 4096 | 26 | sparse_same_definition | 31/200 | 31/200 | 1.000 | no | strict_valid_count |
| 4096 | 27 | dense_local_null | 80/200 | 77/200 | 0.963 | no | strict_valid_count |
| 4096 | 27 | dense_optimized | 77/200 | 77/200 | 1.000 | no | strict_valid_count |
| 4096 | 27 | sparse_same_definition | 21/200 | 21/200 | 1.000 | no | strict_valid_count |
| 4096 | 28 | dense_local_null | 61/200 | 59/200 | 0.967 | no | strict_valid_count |
| 4096 | 28 | dense_optimized | 59/200 | 59/200 | 1.000 | no | strict_valid_count |
| 4096 | 28 | sparse_same_definition | 25/200 | 25/200 | 1.000 | no | strict_valid_count |
| 4096 | 29 | dense_local_null | 151/200 | 147/200 | 0.974 | no | strict_valid_count |
| 4096 | 29 | dense_optimized | 147/200 | 147/200 | 1.000 | no | strict_valid_count |
| 4096 | 29 | sparse_same_definition | 23/200 | 23/200 | 1.000 | no | strict_valid_count |
| 8192 | 23 | dense_local_null | 11/200 | 9/200 | 0.818 | no | strict_valid_count, naturality_valid_fraction |
| 8192 | 23 | dense_optimized | 9/200 | 9/200 | 1.000 | no | strict_valid_count |
| 8192 | 23 | sparse_same_definition | 28/200 | 28/200 | 1.000 | no | strict_valid_count |
| 8192 | 24 | dense_local_null | 52/200 | 47/200 | 0.904 | no | strict_valid_count, naturality_valid_fraction |
| 8192 | 24 | dense_optimized | 47/200 | 47/200 | 1.000 | no | strict_valid_count |
| 8192 | 24 | sparse_same_definition | 30/200 | 30/200 | 1.000 | no | strict_valid_count |
| 8192 | 25 | dense_local_null | 45/200 | 42/200 | 0.933 | no | strict_valid_count, naturality_valid_fraction |
| 8192 | 25 | dense_optimized | 42/200 | 42/200 | 1.000 | no | strict_valid_count |
| 8192 | 25 | sparse_same_definition | 25/200 | 25/200 | 1.000 | no | strict_valid_count |
| 8192 | 26 | dense_local_null | 71/200 | 68/200 | 0.958 | no | strict_valid_count |
| 8192 | 26 | dense_optimized | 68/200 | 68/200 | 1.000 | no | strict_valid_count |
| 8192 | 26 | sparse_same_definition | 30/200 | 30/200 | 1.000 | no | strict_valid_count |
| 8192 | 27 | dense_local_null | 150/200 | 144/200 | 0.960 | no | strict_valid_count |
| 8192 | 27 | dense_optimized | 144/200 | 144/200 | 1.000 | no | strict_valid_count |
| 8192 | 27 | sparse_same_definition | 32/200 | 32/200 | 1.000 | no | strict_valid_count |
| 8192 | 28 | dense_local_null | 66/200 | 62/200 | 0.939 | no | strict_valid_count, naturality_valid_fraction |
| 8192 | 28 | dense_optimized | 62/200 | 62/200 | 1.000 | no | strict_valid_count |
| 8192 | 28 | sparse_same_definition | 29/200 | 29/200 | 1.000 | no | strict_valid_count |
| 8192 | 29 | dense_local_null | 191/200 | 186/200 | 0.974 | yes | - |
| 8192 | 29 | dense_optimized | 186/200 | 186/200 | 1.000 | yes | - |
| 8192 | 29 | sparse_same_definition | 13/200 | 13/200 | 1.000 | no | strict_valid_count |
| 16384 | 23 | dense_local_null | 72/200 | 69/200 | 0.958 | no | strict_valid_count |
| 16384 | 23 | dense_optimized | 69/200 | 69/200 | 1.000 | no | strict_valid_count |
| 16384 | 23 | sparse_same_definition | 15/200 | 15/200 | 1.000 | no | strict_valid_count |
| 16384 | 24 | dense_local_null | 143/200 | 133/200 | 0.930 | no | strict_valid_count, naturality_valid_fraction |
| 16384 | 24 | dense_optimized | 133/200 | 133/200 | 1.000 | no | strict_valid_count |
| 16384 | 24 | sparse_same_definition | 15/200 | 15/200 | 1.000 | no | strict_valid_count |
| 16384 | 25 | dense_local_null | 136/200 | 129/200 | 0.949 | no | strict_valid_count, naturality_valid_fraction |
| 16384 | 25 | dense_optimized | 129/200 | 129/200 | 1.000 | no | strict_valid_count |
| 16384 | 25 | sparse_same_definition | 24/200 | 24/200 | 1.000 | no | strict_valid_count |
| 16384 | 26 | dense_local_null | 120/200 | 113/200 | 0.942 | no | strict_valid_count, naturality_valid_fraction |
| 16384 | 26 | dense_optimized | 113/200 | 113/200 | 1.000 | no | strict_valid_count |
| 16384 | 26 | sparse_same_definition | 11/200 | 11/200 | 1.000 | no | strict_valid_count |
| 16384 | 27 | dense_local_null | 134/200 | 127/200 | 0.948 | no | strict_valid_count, naturality_valid_fraction |
| 16384 | 27 | dense_optimized | 127/200 | 127/200 | 1.000 | no | strict_valid_count |
| 16384 | 27 | sparse_same_definition | 19/200 | 19/200 | 1.000 | no | strict_valid_count |
| 16384 | 28 | dense_local_null | 19/200 | 19/200 | 1.000 | no | strict_valid_count |
| 16384 | 28 | dense_optimized | 19/200 | 19/200 | 1.000 | no | strict_valid_count |
| 16384 | 28 | sparse_same_definition | 23/200 | 23/200 | 1.000 | no | strict_valid_count |
| 16384 | 29 | dense_local_null | 172/200 | 167/200 | 0.971 | yes | - |
| 16384 | 29 | dense_optimized | 167/200 | 167/200 | 1.000 | yes | - |
| 16384 | 29 | sparse_same_definition | 27/200 | 27/200 | 1.000 | no | strict_valid_count |
