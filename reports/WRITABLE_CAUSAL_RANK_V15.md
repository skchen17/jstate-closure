# Restricted writable finite-response rank — V15

Actual finite-writeback response columns were measured at ε=1 in the **same V13 probe-coordinate domain** as frozen ideal JVP columns. Probe counts 64/128/256/512 were measured at one train anchor; 64 columns were measured at each of four additional train anchors. Each block is train-normalized before the stacked spectrum. This is not a full raw model-state intrinsic dimension.

Stacked-normalized spectra (joint channels):

| base_trial_id                  | operator         |   direction_count |   rank_90 |   rank_95 |   rank_99 |   stable_rank |   effective_rank |
|:-------------------------------|:-----------------|------------------:|----------:|----------:|----------:|--------------:|-----------------:|
| v13-train-90d330b478659e3d0bc3 | finite_writeback |                64 |         6 |         8 |        13 |          2.22 |             5.32 |
| v13-train-90d330b478659e3d0bc3 | autograd_jvp     |                64 |         4 |         5 |         8 |          2.12 |             3.98 |
| v13-train-90d330b478659e3d0bc3 | finite_writeback |               128 |         7 |        10 |        18 |          2.17 |             6.09 |
| v13-train-90d330b478659e3d0bc3 | autograd_jvp     |               128 |         5 |         6 |        11 |          2.10 |             4.52 |
| v13-train-90d330b478659e3d0bc3 | finite_writeback |               256 |         8 |        12 |        24 |          2.19 |             6.72 |
| v13-train-90d330b478659e3d0bc3 | autograd_jvp     |               256 |         5 |         7 |        14 |          2.12 |             4.80 |
| v13-train-90d330b478659e3d0bc3 | finite_writeback |               512 |         9 |        14 |        30 |          2.20 |             7.20 |
| v13-train-90d330b478659e3d0bc3 | autograd_jvp     |               512 |         5 |         7 |        17 |          2.12 |             5.01 |
| v13-train-6bb01e7e53b688215c6e | finite_writeback |                64 |         7 |        10 |        16 |          2.81 |             7.04 |
| v13-train-6bb01e7e53b688215c6e | autograd_jvp     |                64 |         4 |         5 |         9 |          2.22 |             4.43 |
| v13-train-4e63d69f5c3f40704813 | finite_writeback |                64 |         7 |         9 |        14 |          3.02 |             7.10 |
| v13-train-4e63d69f5c3f40704813 | autograd_jvp     |                64 |         4 |         5 |         8 |          2.04 |             4.10 |
| v13-train-c6eec494888d6d518e28 | finite_writeback |                64 |         7 |        10 |        15 |          3.51 |             8.24 |
| v13-train-c6eec494888d6d518e28 | autograd_jvp     |                64 |         4 |         5 |         8 |          2.07 |             4.52 |
| v13-train-07385890263e531ff8e3 | finite_writeback |                64 |         7 |         9 |        14 |          2.38 |             6.46 |
| v13-train-07385890263e531ff8e3 | autograd_jvp     |                64 |         5 |         6 |         9 |          2.02 |             4.46 |

The signal-qualified subset uses both signed J effects above the frozen `0.00680280` threshold. Counts and restricted spectra:

| base_trial_id                  |   qualified_direction_count | rank_identifiable   |   r95_R |   r95_J |
|:-------------------------------|----------------------------:|:--------------------|--------:|--------:|
| v13-train-90d330b478659e3d0bc3 |                         294 | True                |      13 |       7 |
| v13-train-6bb01e7e53b688215c6e |                          39 | True                |       8 |       5 |
| v13-train-4e63d69f5c3f40704813 |                          30 | True                |       9 |       5 |
| v13-train-c6eec494888d6d518e28 |                          27 | True                |       9 |       5 |
| v13-train-07385890263e531ff8e3 |                          30 | True                |       9 |       6 |

At the 512-probe anchor, stacked `r95_R=14` versus `r95_J=7`; the five 64-probe anchors give `r95_R=8–10` versus `r95_J=5–6`. This supports the **restricted finite-response matrix** pattern in V15-D (`FINITE_WRITABLE_CAUSAL_OPERATOR_HIGHER_RANK_THAN_IDEAL_DIFFERENTIAL_OPERATOR`) but not a linear writable operator theorem. Full numerical spectra can include weak columns and quantization/readout noise; qualified-subset spectra have selection bias. A numeric `r95_R` alone does **not** establish `LOW_DIMENSIONAL_WRITABLE_CAUSAL_ACTUATOR_SUPPORTED` while the finite-response linearity gate fails. Complete singular vectors are reproducible from the response-vector list columns in `finite_operator_columns_v15.parquet`; the spectrum JSON preserves singular values.
