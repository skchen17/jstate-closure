# V23 Complete Report

## Identity

- Name: **Oracle Local Action Charts and Input-Rank Scaling**
- Parent: `c755b4d7b0baf3aa0291456fcf49169087b8c863`
- Protocol hash: `378297e7c0413b1bcdddc05894dbcaff23835cdb349567dceb73ebb8bd387c66`
- State split hash: `762f62fee5c470ab719983cbad34b5822a2dde33dc130a33101cb1b9f48c035b`
- Probe hashes: `{"128": "e1daff6b0e3ac35c47e3c4d4780a55211f4250aca98b14a530fd6ab08083c450", "256": "24d3d90c2dc3fe7cac33f625789b1d855d89eeb5a5fbf1a76861288a83dbce03", "512": "9a1dd253da8e52b3163942f1a22bd2a65573e34522f7654782f7a34b05f48804", "64": "1dbe804f7b4c3e0c7cd12449c7ce760c3381a0d7a9689c05a9077275052e722b"}`
- Train / held-out / new-final hashes: `9a1dd253da8e52b3163942f1a22bd2a65573e34522f7654782f7a34b05f48804` / `793cab02928568a7201c863993943c76da566809bb51fb6ce7f573e91fa14d5f` / `df6f69a7b5b07f8af9ec1e460d6bb63f3ffe61cc79a3c92812d4ed552a1e2b2b`

## Outcome

Formal outcome: **V23-E_INPUT_RANK_NOT_SATURATED+V23-F_ORACLE_CHART_INSUFFICIENT**.

The tested low-rank subspace chart is insufficient for held-out finite-action prediction; this does not imply that local charts do not exist. Input-sensitive causal rank remains probe-limited if saturation fails.

## Rank scaling

| m | Gram rank | condition | JVP r90/r95/r99 | finite r90/r95/r99 |
|---:|---:|---:|---:|---:|
| 64 | 64 | 9.046e+04 | [11.0, 17.0, 31.5] | [15.5, 21.0, 33.0] |
| 128 | 128 | 7.690e+05 | [15.0, 23.0, 48.0] | [18.5, 27.0, 46.5] |
| 256 | 254 | 2.979e+06 | [17.5, 28.0, 65.0] | [23.0, 33.5, 60.5] |
| 512 | 417 | 5.413e+06 | [19.0, 31.0, 75.0] | [25.0, 36.5, 67.5] |

Joint rank saturation: **FALSE**.

## Oracle charts

Best JVP: `k24:M4_cubic_nonlinear`; best finite: `k24:M0_nearest`; `k_chart_min=N/A`.

### Best finite metrics

| category | relative L2 | cosine | J cosine | norm ratio |
|---|---:|---:|---:|---:|
| unseen_amplitude | 0.960954 | 0.317000 | 0.310984 | 0.201072 |
| unseen_dense | 0.769178 | 0.597264 | 0.570696 | 0.715100 |
| unseen_direction | 1.025821 | 0.377751 | 0.338953 | 0.727023 |
| unseen_pair | 0.118783 | 0.806665 | 0.880796 | 0.996332 |
| unseen_sign | 1.019546 | 0.393395 | 0.344085 | 0.724646 |

### Best JVP metrics

| category | relative L2 | cosine | J cosine | norm ratio |
|---|---:|---:|---:|---:|
| unseen_amplitude | 0.878402 | 0.646534 | 0.626799 | 0.207469 |
| unseen_dense | 0.452086 | 0.866666 | 0.843087 | 0.836195 |
| unseen_direction | 0.843700 | 0.662272 | 0.683380 | 0.680571 |
| unseen_pair | 0.113035 | 0.811872 | 0.856977 | 0.876703 |
| unseen_sign | 0.827548 | 0.653629 | 0.672629 | 0.690202 |

JVP/finite input overlap: `0.359687`; angle: `55.258445°`.

## Smoothness, aliasing, transport

- Nearest-J clean-state chart angle: `51.313420°`; adjacent-token result unavailable on the frozen panel.
- Same-J P0/Pq chart angle: `51.726420°`.
- Procrustes transport fidelity: `0.337425`.

## Prediction and adaptation

- Chart-predictor status: `NOT_RUN_ORACLE_GATE_FAILED`; S0/S1/S2/S4 = `N/A/N/A/N/A/N/A`.
- Local probe curve: `{"128": {"best_token": "k32:M0_nearest", "median_cosine": 0.38607414960140807, "median_j_cosine": 0.348008725962357, "median_norm_ratio": 0.7142199043858382, "relative_l2": 1.019415099502293, "sample_count": 3200}, "16": {"best_token": "k12:M0_nearest", "median_cosine": 0.35119735147429276, "median_j_cosine": 0.3351858789811696, "median_norm_ratio": 0.7528161745833588, "relative_l2": 1.013987146551028, "sample_count": 3200}, "32": {"best_token": "k32:M0_nearest", "median_cosine": 0.363111439163039, "median_j_cosine": 0.3862278914622397, "median_norm_ratio": 0.8187013260238863, "relative_l2": 1.0223800648995869, "sample_count": 3200}, "4": {"best_token": "k4:M2_ridge", "median_cosine": 0.4979512772478455, "median_j_cosine": 0.41367253496208634, "median_norm_ratio": 0.6790563808784552, "relative_l2": 0.9264408380315695, "sample_count": 3200}, "64": {"best_token": "k64:M0_nearest", "median_cosine": 0.37766823683066963, "median_j_cosine": 0.39511790983697914, "median_norm_ratio": 0.8210619182045444, "relative_l2": 1.0152519252826655, "sample_count": 3200}, "8": {"best_token": "k8:M2_ridge", "median_cosine": 0.5194881231497745, "median_j_cosine": 0.4695532470665056, "median_norm_ratio": 0.6793416466948523, "relative_l2": 0.9312359388926131, "sample_count": 3200}}`.
- Oracle vs learned direction L2: `0.843700` vs `1.001623`.

## Authorization

- `COMPACT_OPERATOR_SEARCH_REOPENED = FALSE`
- `RAW_TO_OPERATOR_ENCODER_AUTHORIZED = FALSE`
- `H2_REMAINS = TRUE`
- `H3_AUTHORIZED = FALSE`
- `DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`
- Independent final: `FROZEN_AND_UNOPENED_NO_DEVELOPMENT_VALIDATION_FINALIST`

## Verification

V23 tests: `6 passed`. Full suite: `244 passed, 2 inherited failures`. Historical records were not rewritten to hide inherited failures.

## Files

All required standalone V23 reports are in `reports/`; machine-readable JSON/Parquet outputs are in `results/v23/processed/`; raw response matrices remain in the external V23 scratch directory.
