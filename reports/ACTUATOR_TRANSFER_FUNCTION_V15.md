# Raw persistent-state actuator transfer — V15

Five frozen V13 train anchors × five predeclared probe directions were audited at 13 absolute scales and six state-relative ULP scales under seven precision/storage modes. The actual V13 `_apply_direction` writeback was read back tensor by tensor for REC, Conv, attention K and V at every touched layer. This is a **diagnostic train** measurement, not a causal control result. `ACTUATOR_DEADZONE_MAX_TESTED_SCALE` is the largest tested absolute scale with median exactly-zero fraction ≥0.5; it is not a physical discontinuity at that precise scale. `MIN_STATE_CHANGE_SCALE` is the first tested scale with any surviving element; `MIN_RELIABLE_STATE_SCALE` additionally requires median gain in [0.8,1.2] and cosine ≥0.95.

| Channel | Dead-zone max tested | First state change | First reliable state scale | Gain at scale 1 | Surviving fraction at scale 1 |
|---|---:|---:|---:|---:|---:|
| conv | 0.1 | 3e-05 | 0.25 | 1.0011 | 0.934 |
| keys | 0.03 | 0.0001 | 0.5 | 1.0060 | 0.730 |
| recurrent | 0.01 | 1e-05 | 0.25 | 1.0021 | 0.878 |
| values | 0.03 | 0.0003 | 0.5 | 1.0036 | 0.682 |

The per-layer requested/realized norm, cosine, exact-zero fraction, sign flips, relative norm loss, max error, effective BF16 ULP, below-one/half-ULP fraction and surviving fraction are in `results/v15/processed/actuator_transfer_v15.parquet`. The local ULP sweeps and all seven modes are in the same Parquet; per-channel/layer curves are in `actuator_transfer_corrected_v15.json`.

Absolute-scale native curves (median across measured layer/directions):

| channel   |   scale |   median_gain |   median_cosine |   median_zero_fraction |   median_surviving_fraction |
|:----------|--------:|--------------:|----------------:|-----------------------:|----------------------------:|
| conv      |   0.001 |         0.131 |           0.096 |                  0.993 |                       0.007 |
| conv      |   0.010 |         0.401 |           0.305 |                  0.935 |                       0.065 |
| conv      |   0.100 |         1.012 |           0.798 |                  0.507 |                       0.493 |
| conv      |   0.250 |         1.016 |           0.954 |                  0.249 |                       0.751 |
| conv      |   0.500 |         1.010 |           0.988 |                  0.132 |                       0.868 |
| conv      |   1.000 |         1.001 |           0.997 |                  0.066 |                       0.934 |
| conv      |   2.000 |         1.001 |           0.999 |                  0.033 |                       0.967 |
| keys      |   0.001 |         0.113 |           0.094 |                  0.990 |                       0.002 |
| keys      |   0.010 |         0.396 |           0.290 |                  0.912 |                       0.019 |
| keys      |   0.100 |         0.955 |           0.758 |                  0.479 |                       0.164 |
| keys      |   0.250 |         1.013 |           0.922 |                  0.248 |                       0.346 |
| keys      |   0.500 |         1.019 |           0.974 |                  0.134 |                       0.547 |
| keys      |   1.000 |         1.006 |           0.993 |                  0.066 |                       0.730 |
| keys      |   2.000 |         1.002 |           0.998 |                  0.035 |                       0.849 |
| recurrent |   0.001 |         0.205 |           0.156 |                  0.970 |                       0.006 |
| recurrent |   0.010 |         0.616 |           0.468 |                  0.745 |                       0.055 |
| recurrent |   0.100 |         1.009 |           0.908 |                  0.211 |                       0.397 |
| recurrent |   0.250 |         1.010 |           0.974 |                  0.095 |                       0.633 |
| recurrent |   0.500 |         1.005 |           0.992 |                  0.049 |                       0.778 |
| recurrent |   1.000 |         1.002 |           0.998 |                  0.025 |                       0.878 |
| recurrent |   2.000 |         1.000 |           0.999 |                  0.012 |                       0.937 |
| values    |   0.001 |         0.122 |           0.093 |                  0.988 |                       0.002 |
| values    |   0.010 |         0.434 |           0.343 |                  0.882 |                       0.023 |
| values    |   0.100 |         0.954 |           0.812 |                  0.378 |                       0.201 |
| values    |   0.250 |         1.007 |           0.920 |                  0.203 |                       0.378 |
| values    |   0.500 |         1.003 |           0.967 |                  0.116 |                       0.532 |
| values    |   1.000 |         1.004 |           0.991 |                  0.060 |                       0.682 |
| values    |   2.000 |         1.003 |           0.997 |                  0.028 |                       0.808 |

At small scales Conv has the largest median zero-writeback region by the declared 50% rule; at scale 1 V survives in the smallest fraction of nonzero requested elements. These are different definitions of “largest quantization loss.” The initial summary `actuator_transfer_v15.json` is preserved: its empty curves were caused by using the pandas `DataFrame.mode` method instead of the `mode` column. The separate frozen correction recomputed only aggregates, not measured rows (digest `53fb6908ae059344fad21037d2707d0ce861d6e730118f00857a0d4a6e3205dd`).

The FP32-shadow experiment accumulates repeated requests in FP32 and performs one BF16 cast immediately before hypothetical model consumption. For a single final write this equals cast-after-add of the total delta. It is compared with repeated BF16 state writes without changing model weights or forward consumption semantics; only raw realized state, not model output, was measured:

| mode                            | channel   |   increment |   steps |   median_gain |   median_cosine |   median_surviving_fraction |
|:--------------------------------|:----------|------------:|--------:|--------------:|----------------:|----------------------------:|
| fp32_shadow_cast_at_consumption | conv      |       0.010 |       1 |         0.401 |           0.305 |                       0.065 |
| fp32_shadow_cast_at_consumption | conv      |       0.010 |       4 |         0.751 |           0.572 |                       0.244 |
| fp32_shadow_cast_at_consumption | conv      |       0.010 |      16 |         1.010 |           0.900 |                       0.637 |
| fp32_shadow_cast_at_consumption | keys      |       0.010 |       1 |         0.396 |           0.290 |                       0.019 |
| fp32_shadow_cast_at_consumption | keys      |       0.010 |       4 |         0.695 |           0.552 |                       0.069 |
| fp32_shadow_cast_at_consumption | keys      |       0.010 |      16 |         0.994 |           0.852 |                       0.247 |
| fp32_shadow_cast_at_consumption | recurrent |       0.010 |       1 |         0.616 |           0.468 |                       0.055 |
| fp32_shadow_cast_at_consumption | recurrent |       0.010 |       4 |         0.937 |           0.766 |                       0.199 |
| fp32_shadow_cast_at_consumption | recurrent |       0.010 |      16 |         1.009 |           0.951 |                       0.519 |
| fp32_shadow_cast_at_consumption | values    |       0.010 |       1 |         0.434 |           0.343 |                       0.023 |
| fp32_shadow_cast_at_consumption | values    |       0.010 |       4 |         0.823 |           0.644 |                       0.091 |
| fp32_shadow_cast_at_consumption | values    |       0.010 |      16 |         0.986 |           0.878 |                       0.289 |
| repeated_bf16_storage_write     | conv      |       0.010 |       1 |         0.401 |           0.305 |                       0.065 |
| repeated_bf16_storage_write     | conv      |       0.010 |       4 |         0.398 |           0.305 |                       0.065 |
| repeated_bf16_storage_write     | conv      |       0.010 |      16 |         0.394 |           0.304 |                       0.065 |
| repeated_bf16_storage_write     | keys      |       0.010 |       1 |         0.396 |           0.290 |                       0.019 |
| repeated_bf16_storage_write     | keys      |       0.010 |       4 |         0.396 |           0.290 |                       0.019 |
| repeated_bf16_storage_write     | keys      |       0.010 |      16 |         0.390 |           0.288 |                       0.019 |
| repeated_bf16_storage_write     | recurrent |       0.010 |       1 |         0.616 |           0.468 |                       0.055 |
| repeated_bf16_storage_write     | recurrent |       0.010 |       4 |         0.611 |           0.468 |                       0.055 |
| repeated_bf16_storage_write     | recurrent |       0.010 |      16 |         0.599 |           0.467 |                       0.055 |
| repeated_bf16_storage_write     | values    |       0.010 |       1 |         0.434 |           0.343 |                       0.023 |
| repeated_bf16_storage_write     | values    |       0.010 |       4 |         0.434 |           0.343 |                       0.023 |
| repeated_bf16_storage_write     | values    |       0.010 |      16 |         0.431 |           0.342 |                       0.023 |
