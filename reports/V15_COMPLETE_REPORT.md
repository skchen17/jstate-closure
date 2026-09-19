# V15 complete report

> This is the canonical single-file bundle for this version. It combines the 
> version-specific FINAL_REPORT section, every standalone version report, and 
> an integrity index of the machine-readable records. Standalone reports remain 
> preserved for direct navigation.

## Bundle provenance

- Source commit: `936db401274c549c126a0588052cd62fb9753715`
- Generated at: `2026-09-19T14:44:58.637496+00:00`
- Included standalone reports: `9`
- Indexed machine-record files: `36`
- Generator: `scripts/build_complete_version_report.py`

## Version summary and adjudication

## V15 — Quantization-Aware Causal Actuation

Formal status: **V15-STOP — FINITE_RESPONSE_LINEARITY_GATE_FAILED**. Real BF16 writeback dead-zones were quantified per REC/Conv/K/V; only 16/100 adaptive state×direction×channel combinations met the predeclared finite-effect selection rule. At accurate realized state perturbations, some output targets still disagree with frozen exact JVP, supporting tested-regime V15-B. The restricted finite-response matrix has higher r95 than ideal JVP (V15-D pattern), but the frozen finite-response linearity gate failed, so spectra and actuator-aware closed-loop results remain diagnostic/development rather than validated finite control. No eligible finalist or new independent confirmatory bank was created. H2 remains; H3 and absolute state replacement are not supported; autonomous-controller training is not authorized. The required cumulative-report append invalidates one old V14 whole-file integrity test; V15's guard confirms all other historical bytes are unchanged. See `reports/V15_COMPLETE_REPORT.md` for all standalone reports, machine records, frozen gates and limitations.

## Standalone report integrity index

| report | SHA256 |
|---|---|
| `reports/ACTUATOR_TRANSFER_FUNCTION_V15.md` | `1b2148787b5e7ae7ae05844d6a0fd42e2ae26a8f4e80b5ce3b9e926aac2839e7` |
| `reports/AUTOGRAD_VS_ACTUATOR_OPERATOR_V15.md` | `5440bd7238ffe6be6858942eb0df11b62b5a936b2d42d00b0dd8df91fca26db7` |
| `reports/CHANNEL_ACTUATOR_GEOMETRY_V15.md` | `595f60a5084d536df1f03e2fce1ca2c03404b9060be92c61be4bb5a6bd4ccd70` |
| `reports/EXECUTION_MANIFEST_V15.md` | `d5c86f20d7b1270e0438f2d928a6952b9111b54d3ed3213c5c1cf79b53d5d1ae` |
| `reports/FINITE_CAUSAL_CONTROLLABILITY_V15.md` | `9323eb7b98841918a0a5c97f9ea2d0220871b6d58977060f277824f71a11285d` |
| `reports/FINITE_CAUSAL_CONTROL_V15.md` | `252e0a21c6d656cd19245139f0cbd4820d476de2333f1d4b4523849ff4980c88` |
| `reports/FINITE_RESPONSE_LINEARITY_V15.md` | `6615dda60f742427f1944eeabf4c90082046d928d226e6df4fbf5db74ab1ae72` |
| `reports/STRICT_INTERFACE_AUDIT_V15.md` | `d5957a63198bb283c6ecd789c2f40d2d312dc58f23d176906034667be48c5a03` |
| `reports/WRITABLE_CAUSAL_RANK_V15.md` | `0a7da87b51bbee1b5cf3a1e0cedc15e0304db29bdee4f1299bf3d0b9a08833dd` |

## Machine-record integrity index

| record | bytes | SHA256 |
|---|---:|---|
| `results/v15/processed/actuator_basis_coefficients_v15.json` | 1622546 | `199bdd92e36b6ba9c55e8919d6f990b95a4391a2bd4115b611622c519370497c` |
| `results/v15/processed/actuator_basis_coverage_v15.parquet` | 9764 | `15da4bba6be3eaca89b791f08572375fd2b3b3cfc02b6b4c6151d1ec3ea3bfdc` |
| `results/v15/processed/actuator_basis_v15.json` | 18187 | `c226e8249ba20cf1f93679331fe851de3c115a831519526b8c7b6f97437622f2` |
| `results/v15/processed/actuator_transfer_corrected_v15.json` | 152308 | `5de7388db4123667f94247ceaa0b342a7ef7ee6ac99d5ce4d6741ad6c58203f9` |
| `results/v15/processed/actuator_transfer_v15.json` | 84485 | `b6ef4af846d2015ce2cd3790b8a55249d9ab41026206a838080619a9857e829e` |
| `results/v15/processed/actuator_transfer_v15.parquet` | 1527088 | `2c719c29467cc6c6e5068e76472d757676d5ec9bb688e326046b4566c5e90cf8` |
| `results/v15/processed/autograd_vs_actuator_v15.json` | 4451 | `5200d8ad2b4f94ee71e49a29459bf16f7e4208b8abbdf82bf39cf469218fc289` |
| `results/v15/processed/autograd_vs_actuator_v15.parquet` | 294180 | `28a645be8151630a8b02c8d60dcfc6735314197add35e85fdb90faf51a29eaae` |
| `results/v15/processed/channel_actuator_geometry_v15.json` | 46491 | `4c63612efd541946432d254cfe00df76960d1bf417dd33770aae8f0e83cb819b` |
| `results/v15/processed/channel_interaction_v15.parquet` | 108379 | `86e34cc1e3b2ae98a0c74e43ba67bf26a00af910467f62f2990f781a62d84afa` |
| `results/v15/processed/control_development_v15.parquet` | 26629 | `a4a80f83c4b1c9a6a44dddb5f24114e0b90c897d65188db44777aab7c42816ae` |
| `results/v15/processed/control_fidelity_v15.parquet` | 27876 | `0c1892364e0926d349be327d192015715e97134472c66a31236ae664dfdf4cb5` |
| `results/v15/processed/control_progress_v15.json` | 49 | `48850e0a59f585cdaa8fa9ac8ad2cd800001b6db1e12f05baa88cbfe314f78d7` |
| `results/v15/processed/control_rank_candidates_v15.parquet` | 6052 | `9b9dc6f1fc6dd4ac981abea565c674c1103a09044a3cb2b209583357f8ba0174` |
| `results/v15/processed/control_terminal_v15.json` | 1361 | `f656f25e7ddf4b3eea6ef9c7e0658ae4c160c78035eb2c8f451f4f67e3cb0b56` |
| `results/v15/processed/control_terminal_v15.parquet` | 8358 | `8915f7a2b0a12e5b3c2dcee34147d8627aa36e83c5a05a78852e00770b81a5b4` |
| `results/v15/processed/finite_causal_control_v15.json` | 7514 | `c3b9caf662151585c4c18cdf133606ea276d36423b510983bc616b6bc68eb594` |
| `results/v15/processed/finite_controllability_v15.parquet` | 8665 | `7112827cc995b3397dee72c87a13242f0ce6f0faa9377567d04df4f0b12ec463` |
| `results/v15/processed/finite_operator_columns_v15.parquet` | 2165274 | `df5fd1a7dd75814773d5bbe41e66164bd769ed5b343bfbb83468ee24a618a87d` |
| `results/v15/processed/finite_operator_snr_v15.parquet` | 88523 | `e50b035a5b36a593111b94c7b2e68b79c523e312b5b1647d21ffc8db3d631349` |
| `results/v15/processed/finite_operator_spectra_v15.json` | 489695 | `89a1555d0c4b5e4f684778fc0312a7325b39ce97667d88351843e18fd6852969` |
| `results/v15/processed/finite_response_linearity_v15.json` | 1499 | `c2e94e7e1fa3715886fd3b5bb746da35ab8eb4975dce8fe1f99a96a138d9f347` |
| `results/v15/processed/finite_response_linearity_v15.parquet` | 8416 | `c53506de141364332373b29548eeec11efcf481ac4b0aad8dacf9f4378b76d65` |
| `results/v15/processed/finite_response_repeat_v15.json` | 388 | `e272c92ab9cf92b9a3afad67e3d3938bdac3608576b90012d2b86f65a048bdf0` |
| `results/v15/processed/finite_response_repeat_v15.parquet` | 6263 | `1a83ecc8c0599a7571ff28569db6612cfbf3b722151fb13e9f876f013c779a99` |
| `results/v15/processed/pilot_progress_v15.json` | 51 | `0f2a0af3e2d3535772793a1367b7cc6ee015ae3cb0bc12af682f0839a255b0c6` |
| `results/v15/processed/precision_modes_v15.json` | 20622 | `dd849619387c963c9712ae4ed8fb69b9cb010d00effcbd721f676b221277482f` |
| `results/v15/processed/precision_modes_v15.parquet` | 26745 | `3210d8ebd8ce4907f93e34b61832d473b0281fdb889ad68669b221bea746df68` |
| `results/v15/processed/precision_progress_v15.json` | 51 | `0f2a0af3e2d3535772793a1367b7cc6ee015ae3cb0bc12af682f0839a255b0c6` |
| `results/v15/processed/rank_progress_v15.json` | 84 | `c826b8b250e817d37ccf9e467e67bcee2adfd771a99da9b52a1a7ad55f8c81d6` |
| `results/v15/processed/shadow_accumulation_v15.json` | 32513 | `fc4ef9cf296042fbc80e47ee5e6d9c0bb7b8d356f0c2e8390c8f0f18e27016c4` |
| `results/v15/processed/shadow_accumulation_v15.parquet` | 404380 | `16bdf27bafda5ff8ec21c8fe0064f06f01beac7880ed6a897cc152a1ae6f7e28` |
| `results/v15/processed/transfer_progress_v15.json` | 51 | `0f2a0af3e2d3535772793a1367b7cc6ee015ae3cb0bc12af682f0839a255b0c6` |
| `results/v15/processed/v15_integrity.json` | 14724 | `f66b6e23f34bb9a3d0367cf9dfb1bb3386af19856a4d1c78f176c8f5672d2551` |
| `results/v15/processed/writable_causal_rank_v15.json` | 798 | `56f08b0403bc276f548d7368264bf54abaa8c09d7dc78a20d4f103c74b5e4527` |
| `results/v15/processed/writable_rank_analysis_v15.json` | 67758 | `337c6abaa5f9a7fdcc43296c7da7aa7c522d498567d5a78f57b58025db8ead12` |

---

## Bundled report 1: `ACTUATOR_TRANSFER_FUNCTION_V15.md`

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

---

## Bundled report 2: `AUTOGRAD_VS_ACTUATOR_OPERATOR_V15.md`

# Ideal autograd JVP versus actual finite actuator — V15

`Jv` is the frozen V13 exact-autograd differential column. `Rεv=[Y(W(P,+εv))-Y(W(P,-εv))]/(2ε)` uses real persistent-state writeback and the frozen J, selected-logit, semantic-continuous and workspace readouts. Target-block normalization uses only V13 **train** JVP columns: `{'j': 0.00637603926805087, 'logits': 0.37930603920841477, 'semantic_continuous': 0.4250155426398896, 'workspace': 0.14939923966331858}`. The fifth target is their dimension-normalized stack. No V13/V14 result was rewritten.

The first epsilon meeting the frozen selection rule was found for only **16/100** train state×direction×channel combinations; unselected trials remain in the Parquet rather than being dropped from the denominator. Selection requires both signed J effects ≥ `0.00680280`, ≥10× measured repeat noise, nonzero state writeback, state gain 0.2–5, and signed-effect norm ratio ≤5. This last check is only a coarse saturation screen, not a proof of local linearity.

Selected comparisons:

| channel   | target              |   count |   cosine |   relative_l2 |   norm_ratio |   realized_state_cosine |
|:----------|:--------------------|--------:|---------:|--------------:|-------------:|------------------------:|
| conv      | j                   |       6 |    0.903 |         0.446 |        1.017 |                   1.000 |
| conv      | logits              |       6 |    0.749 |         0.701 |        1.014 |                   1.000 |
| conv      | semantic_continuous |       6 |    0.748 |         0.692 |        1.057 |                   1.000 |
| conv      | stacked_normalized  |       6 |    0.784 |         0.651 |        1.015 |                   1.000 |
| conv      | workspace           |       6 |    0.717 |         0.753 |        0.953 |                   1.000 |
| joint     | j                   |       7 |    0.997 |         0.080 |        1.025 |                   1.000 |
| joint     | logits              |       7 |    0.984 |         0.194 |        1.037 |                   1.000 |
| joint     | semantic_continuous |       7 |    0.985 |         0.175 |        0.997 |                   1.000 |
| joint     | stacked_normalized  |       7 |    0.981 |         0.196 |        1.011 |                   1.000 |
| joint     | workspace           |       7 |    0.955 |         0.299 |        1.020 |                   1.000 |
| recurrent | j                   |       3 |    0.317 |         0.951 |        0.358 |                   1.000 |
| recurrent | logits              |       3 |    0.199 |         1.017 |        0.565 |                   1.000 |
| recurrent | semantic_continuous |       3 |    0.031 |         1.056 |        0.392 |                   1.000 |
| recurrent | stacked_normalized  |       3 |    0.130 |         1.036 |        0.431 |                   1.000 |
| recurrent | workspace           |       3 |    0.511 |         0.893 |        0.628 |                   1.000 |

Precision decomposition: construction/addition/storage/consumption/readout are explicit fields in `precision_modes_v15.parquet`. `native_fp32_add_bf16_writeback` and `cast_after_add` are the same canonical interface and produce identical outputs; native BF16 add quantizes the increment before addition. FP32 shadow REC/Conv and channel-specific FP32 modes alter diagnostic forward semantics, so they cannot replace the canonical result. Full FP32 KV model consumption is `UNSUPPORTED_BY_FROZEN_MODEL_DTYPE_CONTRACT`; no pretrained weight or query dtype was changed.

At ε=1 (selected diagnostic probes), precision comparisons:

| mode                           | target    | status                                     |   count |   median_cosine |   median_relative_l2 |   median_realized_state_cosine |
|:-------------------------------|:----------|:-------------------------------------------|--------:|----------------:|---------------------:|-------------------------------:|
| cast_after_add                 | j         | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.977 |                0.186 |                          0.998 |
| cast_after_add                 | logits    | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.895 |                0.554 |                          0.998 |
| cast_after_add                 | workspace | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.898 |                0.625 |                          0.998 |
| fp32_conv_only                 | j         | SUPPORTED_DIAGNOSTIC                       |      10 |           0.977 |                0.186 |                          1.000 |
| fp32_conv_only                 | logits    | SUPPORTED_DIAGNOSTIC                       |      10 |           0.895 |                0.554 |                          1.000 |
| fp32_conv_only                 | workspace | SUPPORTED_DIAGNOSTIC                       |      10 |           0.898 |                0.625 |                          1.000 |
| fp32_kv_only                   | j         | UNSUPPORTED_BY_FROZEN_MODEL_DTYPE_CONTRACT |      10 |         nan     |              nan     |                        nan     |
| fp32_kv_only                   | logits    | UNSUPPORTED_BY_FROZEN_MODEL_DTYPE_CONTRACT |      10 |         nan     |              nan     |                        nan     |
| fp32_kv_only                   | workspace | UNSUPPORTED_BY_FROZEN_MODEL_DTYPE_CONTRACT |      10 |         nan     |              nan     |                        nan     |
| fp32_rec_only                  | j         | SUPPORTED_DIAGNOSTIC                       |      10 |           0.981 |                0.205 |                          0.998 |
| fp32_rec_only                  | logits    | SUPPORTED_DIAGNOSTIC                       |      10 |           0.899 |                0.495 |                          0.998 |
| fp32_rec_only                  | workspace | SUPPORTED_DIAGNOSTIC                       |      10 |           0.880 |                0.603 |                          0.998 |
| fp32_shadow_rec_conv           | j         | SUPPORTED_DIAGNOSTIC                       |      10 |           0.981 |                0.205 |                          1.000 |
| fp32_shadow_rec_conv           | logits    | SUPPORTED_DIAGNOSTIC                       |      10 |           0.899 |                0.495 |                          1.000 |
| fp32_shadow_rec_conv           | workspace | SUPPORTED_DIAGNOSTIC                       |      10 |           0.880 |                0.603 |                          1.000 |
| native_bf16_add                | j         | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.977 |                0.186 |                          0.998 |
| native_bf16_add                | logits    | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.895 |                0.554 |                          0.998 |
| native_bf16_add                | workspace | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.898 |                0.625 |                          0.998 |
| native_fp32_add_bf16_writeback | j         | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.977 |                0.186 |                          0.998 |
| native_fp32_add_bf16_writeback | logits    | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.895 |                0.554 |                          0.998 |
| native_fp32_add_bf16_writeback | workspace | SUPPORTED_CANONICAL_WRITEBACK              |      10 |           0.898 |                0.625 |                          0.998 |

At ε=1, canonical joint requested/realized state cosine is near 1 while logits/workspace output mismatch persists. This supports `AUTOGRAD_DIFFERENTIAL_OPERATOR_DOES_NOT_MATCH_FINITE_ACTUATOR_OPERATOR_IN_TESTED_REGIME`; simple BF16 rounding alone does not explain every output mismatch. The direction/target-specific adaptive rows and full signed projections, norm ratios, state realization and ε are in `autograd_vs_actuator_v15.parquet`. Repeatability max relative L2: `0.0`.

---

## Bundled report 3: `CHANNEL_ACTUATOR_GEOMETRY_V15.md`

# Channel-resolved actuator geometry — V15

`R_REC`, `R_CONV`, `R_KV` and `R_joint` were measured at the same ε, state and probe directions; K/V also have separate raw writeback transfer curves. Target-block and stacked spectra are in `finite_operator_spectra_v15.json`.

Stacked-normalized rank at 64 probes across five train anchors:

| base_trial_id                  | channel   |   rank_90 |   rank_95 |   rank_99 |   stable_rank |
|:-------------------------------|:----------|----------:|----------:|----------:|--------------:|
| v13-train-90d330b478659e3d0bc3 | joint     |         6 |         8 |        13 |         2.218 |
| v13-train-90d330b478659e3d0bc3 | recurrent |         4 |         4 |         7 |         2.065 |
| v13-train-90d330b478659e3d0bc3 | conv      |         4 |         5 |         8 |         1.948 |
| v13-train-90d330b478659e3d0bc3 | kv        |         2 |         3 |         5 |         1.245 |
| v13-train-6bb01e7e53b688215c6e | joint     |         7 |        10 |        16 |         2.805 |
| v13-train-6bb01e7e53b688215c6e | recurrent |         3 |         4 |         8 |         2.181 |
| v13-train-6bb01e7e53b688215c6e | conv      |         4 |         5 |         9 |         2.029 |
| v13-train-6bb01e7e53b688215c6e | kv        |         3 |         4 |         6 |         1.589 |
| v13-train-4e63d69f5c3f40704813 | joint     |         7 |         9 |        14 |         3.024 |
| v13-train-4e63d69f5c3f40704813 | recurrent |         3 |         4 |         8 |         1.730 |
| v13-train-4e63d69f5c3f40704813 | conv      |         4 |         6 |        10 |         2.292 |
| v13-train-4e63d69f5c3f40704813 | kv        |         2 |         2 |         3 |         1.146 |
| v13-train-c6eec494888d6d518e28 | joint     |         7 |        10 |        15 |         3.508 |
| v13-train-c6eec494888d6d518e28 | recurrent |         3 |         5 |         8 |         2.146 |
| v13-train-c6eec494888d6d518e28 | conv      |         4 |         6 |        10 |         2.526 |
| v13-train-c6eec494888d6d518e28 | kv        |         2 |         2 |         3 |         1.205 |
| v13-train-07385890263e531ff8e3 | joint     |         7 |         9 |        14 |         2.381 |
| v13-train-07385890263e531ff8e3 | recurrent |         4 |         5 |         8 |         2.212 |
| v13-train-07385890263e531ff8e3 | conv      |         4 |         5 |         8 |         1.783 |
| v13-train-07385890263e531ff8e3 | kv        |         2 |         2 |         4 |         1.891 |

SNR-qualified column fractions: `{'conv': 0.4401041666666667, 'joint': 0.546875, 'kv': 0.01171875, 'recurrent': 0.1640625}`. Median `||R_joint-(R_REC+R_CONV+R_KV)||/||R_joint||` by target: `{'j': 0.10560545330015618, 'logits': 0.1958444811671171, 'semantic_continuous': 0.20340082998229436, 'workspace': 0.297095397911219}`. Interaction is finite-response nonadditivity, not a literal independent-channel decomposition. V14's Conv-removal ablation is preserved; V15 distinguishes its raw actuator dead-zone (larger than REC at small scales) from its actual causal response. Per-direction channel norms and interaction residuals: `channel_interaction_v15.parquet`.
The data support measuring joint REC/Conv/KV effects, but do not prove all three channels are intrinsically necessary: V14's restricted ablation found material Conv removal and no material KV removal under its criterion.

---

## Bundled report 4: `EXECUTION_MANIFEST_V15.md`

# V15 execution manifest

Server workspace: `/data/CSK/J-space-project/jstate-closure` on `222.20.126.223`; parent commit `832d1a57f316a59bd961739f7caea4c4e0c702c8`. GPU stages used `HF_HOME=/data/CSK/J-space-project/.hf-cache` and `/home/user/anaconda3/bin/python`. Exact commands from the repository root:

```bash
python -m jclosure.experiments.actuation_v15 --stage freeze
python -m jclosure.experiments.actuation_v15 --stage raw
python scripts/analyze_actuator_transfer_v15.py
python scripts/freeze_v15_splits.py
python -m jclosure.experiments.operator_v15 --stage pilot
python -m jclosure.experiments.precision_v15
python -m jclosure.experiments.operator_v15 --stage linearity
python -m jclosure.experiments.operator_v15 --stage rank
python scripts/analyze_writable_rank_v15.py
python -m jclosure.experiments.basis_v15
python -m jclosure.experiments.repeat_v15
python -m jclosure.experiments.control_v15
python scripts/analyze_control_v15.py
python -m jclosure.experiments.shadow_v15
python scripts/freeze_v15_finalist_decision.py
python scripts/normalize_v15_numeric_json.py
python -m jclosure.reporting_v15
python scripts/build_v15_integrity.py
python scripts/build_complete_version_report.py V15
python -m pytest -q
python -m pytest -q -k 'not test_v14_integrity_manifest'
```

The first transfer summary had an aggregation-only bug and is preserved; `scripts/analyze_actuator_transfer_v15.py` produced a separate frozen correction. The independent confirmation command is absent because `NO_ELIGIBLE_FINALIST`.

Full `pytest -q` has one expected historical test failure: `test_v14_integrity_manifest` compares the old whole-file SHA256 of the cumulative `reports/FINAL_REPORT.md`, which V15 must append. No V14 hash/test/result was modified. The filtered run excludes only this incompatible test and checks the remaining suite plus V15's explicit parent-byte guard.

| Freeze | Digest | File SHA256 |
|---|---|---|
| `artifacts/quantization_aware_actuation_v15.freeze.json` | `8cccf28e4fd09484e1dd2aaf6dcf2abb311695701e439a40eaee6bdee44f7c35` | `02a16fe8dec376a6eaf88cdee72c0bafc80ff65fc1652c14723c6ca3e2fb713f` |
| `artifacts/quantization_aware_actuation_v15_basis.freeze.json` | `b366c35f00f78a10b9fc70fc10b493d6ade00249036f454a7e6d242b9e198c2f` | `2c8dccd3e1ce8c43ef401f4aec2ca139ae3fce3e51593c6431ae24296b6a85ab` |
| `artifacts/quantization_aware_actuation_v15_control.freeze.json` | `d9df5d03b29c23f95c76022aad2d80e1c51c8e4d8342e3502896d1a9b9d79b54` | `c77193aad84c47cca04fcd7535106a1b315d5cfe13074962744f14dcf87292e2` |
| `artifacts/quantization_aware_actuation_v15_control_terminal_analysis.freeze.json` | `93b239663a1556494165fc7cb9bc142a38b859b0e92865f71eb77d4ba72c8b02` | `57789eb85054dd20941227ee27341f5f4da442b55fd317d6b091a498f10e02a7` |
| `artifacts/quantization_aware_actuation_v15_finalist_decision.freeze.json` | `ab14457731da214771b5b18e775084c1e23cbf633f15e67cb441b944351ee5d9` | `b42f17a1be63baaa2ea0224b26855830192563de54f0cf54e0d788d1a201b18f` |
| `artifacts/quantization_aware_actuation_v15_linearity.freeze.json` | `16b3cc515bdf4224a9a9f0bb818f7dca855318c66606f83b6513b385c21bc12c` | `6eea885fb85a969b266ff0d356998fd26f12d37be5d21b4292d625cce825c8b4` |
| `artifacts/quantization_aware_actuation_v15_pilot.freeze.json` | `a6e84657c6add7c19b6dc8223c5a475f7c6860a112e2ff41c93eefb71365091e` | `3d081d5452671eb13e5c674deb3045a5509a1db574e182a1a25821a6cf83a8e6` |
| `artifacts/quantization_aware_actuation_v15_precision.freeze.json` | `b1410a39973502064a9b8b516830e5c62fe9bc33efb902565a89be52ddea51b7` | `99bb877ae1a8e15e21ae75c4b66b1e38bcce55be24833bb29fcf882dfb424393` |
| `artifacts/quantization_aware_actuation_v15_rank.freeze.json` | `1692a35f4de1db7a64d99ef5cecff6cc0c238a827ab777ba204c4e6083ef737a` | `f845b1cf1fcbdd050e28b2feba4ff3825ab1ec3aa726312cc4112ed5fb341570` |
| `artifacts/quantization_aware_actuation_v15_rank_analysis.freeze.json` | `a6e55f93d6d71c617e549d85c1c8820f152f2d42e39f63bdb1d07f402812fde7` | `ce6073480b609774621b65ee1dd8d68b78b267d5f024e01a0709ff44eb6053ca` |
| `artifacts/quantization_aware_actuation_v15_repeat.freeze.json` | `7190eb64963112c79d1791050c779b00c354ad4f5f248a7b1791568b2d8a795c` | `2af303286347093a8ec69cbf78644bdfaf7b97e7f4a1a75dcadacfb18a21537b` |
| `artifacts/quantization_aware_actuation_v15_shadow_accumulation.freeze.json` | `b295fa97cfbdee0a51c564aea4b1c8f105a85d0d6548bc72111eb890d4c8aae8` | `9ae0c3c8e21473381237406a08c95efe6d22b3e2a54087d850d46bbf2028e424` |
| `artifacts/quantization_aware_actuation_v15_splits.freeze.json` | `a534429fce03c842dc4ead9e102e686754b62689e7a3d2111e88b329203f8716` | `2cef4e7bfdb7869056a96bc82e7c1643f40af150539e30647c272fa2200905bb` |
| `artifacts/quantization_aware_actuation_v15_transfer_summary_correction.freeze.json` | `53fb6908ae059344fad21037d2707d0ce861d6e730118f00857a0d4a6e3205dd` | `e002aef201335f09a34d028b99d6cb97442b78af9a6851eb8103e96ab48f6b2d` |

Machine record hashes are in each JSON summary and `results/v15/processed/v15_integrity.json`. Changed files can be enumerated exactly with `git diff --name-only 832d1a57f316a59bd961739f7caea4c4e0c702c8..HEAD` after commit (or `git status --short` before commit).

---

## Bundled report 5: `FINITE_CAUSAL_CONTROLLABILITY_V15.md`

# Empirical finite causal reachability — V15 development

The one-step columns are local finite-response basis directions. Multi-step columns concatenate the remeasured response maps after attempted writebacks in a common frozen target coordinate system; no full state transition Jacobian is claimed. Projection floors refer only to the tested rank-limited actuator span and weighted teacher residual, not global reachability.

| objective   |   one_step_rank |   multi_step_rank |   one_step_floor |   multi_step_floor |   outside_fraction |
|:------------|----------------:|------------------:|-----------------:|-------------------:|-------------------:|
| h1          |           7.000 |            18.000 |            0.845 |              0.712 |              1.000 |
| h1_h2_h4    |           6.500 |            17.000 |            0.938 |              0.853 |              1.000 |
| h1_h2_h4_h8 |           6.000 |            12.500 |            0.977 |              0.891 |              1.000 |

`TARGET_OUTSIDE_TESTED_ACTUATOR_REACHABLE_SPACE` is recorded per case only when the multi-step projection floor exceeds the predeclared 0.2 threshold. Since the finite linearity gate failed, this is a descriptive development diagnostic, not a formal proof of actuator impossibility. Machine rows: `finite_controllability_v15.parquet`.

---

## Bundled report 6: `FINITE_CAUSAL_CONTROL_V15.md`

# Actuator-calibrated development control — V15

The compact basis is a realized-cost-weighted SVD of **finite response columns**, not PCA or raw autograd singular vectors. The denominator is a diagonal approximation using each column's measured realized-state norm; raw-state cross-Gram and nonlinear combination writeback are not assumed away. The basis coefficients and held-out-probe target-coverage comparison with an autograd-JVP target basis are machine-readable. **The finite basis covers less held-out-probe target energy than the autograd basis in the tested 512-probe anchor**; no basis superiority is claimed.

|   rank |   r95_weighted |   finite_basis_heldout_target_coverage |   autograd_basis_heldout_finite_target_coverage |
|-------:|---------------:|---------------------------------------:|------------------------------------------------:|
|  4.000 |         22.000 |                                  0.283 |                                           0.628 |
|  8.000 |         22.000 |                                  0.428 |                                           0.727 |
| 16.000 |         22.000 |                                  0.664 |                                           0.842 |
| 22.000 |         22.000 |                                  0.763 |                                           0.872 |
| 32.000 |         22.000 |                                  0.849 |                                           0.907 |
| 64.000 |         22.000 |                                  0.964 |                                           0.982 |

On the same ten V13 validation cases, the V15 controller used actual ε=1 finite responses, train/frozen output scales, h1 / h1+h2+h4 / h1+h2+h4+h8 objectives, at most four steps, rank `min(8, local r95)`, and realized-response trust decisions. Every step records requested control norm, realized state norm, channel gains, prediction error, actual residual, acceptance and trust ratio. A teacher raw cache was used **only** to generate the teacher response label, never as candidate input or basis. These coordinates are numerical actuator controls, not cognitive or biological state variables.

Median residual curves:

| method                          |   step |   raw_ratio |   weighted_ratio |   acceptance |
|:--------------------------------|-------:|------------:|-----------------:|-------------:|
| actuator_calibrated_h1          |      1 |       0.937 |            0.933 |        0.800 |
| actuator_calibrated_h1          |      2 |       0.806 |            0.826 |        0.750 |
| actuator_calibrated_h1          |      3 |       0.806 |            0.826 |        0.167 |
| actuator_calibrated_h1          |      4 |       0.850 |            0.878 |        1.000 |
| actuator_calibrated_h1_h2_h4    |      1 |       0.955 |            0.978 |        0.600 |
| actuator_calibrated_h1_h2_h4    |      2 |       0.829 |            0.845 |        0.833 |
| actuator_calibrated_h1_h2_h4    |      3 |       0.738 |            0.834 |        0.600 |
| actuator_calibrated_h1_h2_h4    |      4 |       0.667 |            0.801 |        0.333 |
| actuator_calibrated_h1_h2_h4_h8 |      1 |       0.919 |            0.990 |        0.600 |
| actuator_calibrated_h1_h2_h4_h8 |      2 |       0.751 |            0.893 |        0.833 |
| actuator_calibrated_h1_h2_h4_h8 |      3 |       0.748 |            0.897 |        0.400 |
| actuator_calibrated_h1_h2_h4_h8 |      4 |       0.655 |            0.829 |        1.000 |

Those stepwise medians have different surviving case sets and need not be monotone even when a within-case accepted step improves the weighted objective. The ten-case terminal analysis includes stopped cases:

| method                          |   case_count |   terminal_raw_ratio |   terminal_weighted_ratio |   raw_monotone_fraction |   weighted_monotone_fraction |   terminal_improvement_fraction |
|:--------------------------------|-------------:|---------------------:|--------------------------:|------------------------:|-----------------------------:|--------------------------------:|
| actuator_calibrated_h1          |           10 |                0.854 |                     0.863 |                   0.900 |                        1.000 |                           0.800 |
| actuator_calibrated_h1_h2_h4    |           10 |                0.899 |                     0.902 |                   0.800 |                        1.000 |                           0.600 |
| actuator_calibrated_h1_h2_h4_h8 |           10 |                0.835 |                     0.911 |                   0.800 |                        1.000 |                           0.600 |

Weighted residual is monotone by acceptance construction, but raw teacher residual was not monotone for every case and terminal improvement was not universal.

Same development-panel controller comparison (raw teacher is the target label, so its self-fidelity is trivially 1, not a competing learned controller):

| method                                  |   horizon |   direction |   magnitude |   output |   semantic_legacy |   semantic_continuous |   sign |
|:----------------------------------------|----------:|------------:|------------:|---------:|------------------:|----------------------:|-------:|
| actuator_calibrated_h1                  |         1 |       0.290 |       0.307 |    0.308 |             0.230 |                 0.232 |  0.400 |
| actuator_calibrated_h1                  |         2 |       0.303 |       0.496 |    0.317 |             0.020 |                 0.243 |  0.600 |
| actuator_calibrated_h1                  |         4 |       0.453 |       0.706 |    0.472 |             0.010 |                 0.363 |  0.500 |
| actuator_calibrated_h1                  |         8 |       0.508 |       0.777 |    0.543 |             0.070 |                 0.406 |  0.400 |
| actuator_calibrated_h1_h2_h4            |         1 |       0.126 |       0.234 |    0.301 |             0.020 |                 0.075 |  0.200 |
| actuator_calibrated_h1_h2_h4            |         2 |       0.257 |       0.413 |    0.327 |             0.040 |                 0.154 |  0.400 |
| actuator_calibrated_h1_h2_h4            |         4 |       0.436 |       0.555 |    0.517 |             0.020 |                 0.261 |  0.500 |
| actuator_calibrated_h1_h2_h4            |         8 |       0.528 |       0.565 |    0.541 |             0.050 |                 0.317 |  0.500 |
| actuator_calibrated_h1_h2_h4_h8         |         1 |       0.190 |       0.218 |    0.308 |             0.070 |                 0.114 |  0.200 |
| actuator_calibrated_h1_h2_h4_h8         |         2 |       0.276 |       0.383 |    0.309 |             0.030 |                 0.166 |  0.600 |
| actuator_calibrated_h1_h2_h4_h8         |         4 |       0.465 |       0.542 |    0.488 |             0.030 |                 0.279 |  0.500 |
| actuator_calibrated_h1_h2_h4_h8         |         8 |       0.526 |       0.548 |    0.610 |             0.040 |                 0.316 |  0.500 |
| closed_loop_finite_response_h1          |         1 |       0.699 |       0.981 |    0.671 |             0.430 |                 0.699 |  0.800 |
| closed_loop_finite_response_h1          |         2 |       0.535 |       1.005 |    0.514 |             0.110 |                 0.535 |  0.800 |
| closed_loop_finite_response_h1          |         4 |       0.498 |       1.000 |    0.537 |             0.070 |                 0.498 |  0.800 |
| closed_loop_finite_response_h1          |         8 |       0.511 |       1.008 |    0.491 |             0.120 |                 0.511 |  0.900 |
| closed_loop_finite_response_h1_h2_h4_h8 |         1 |       0.688 |       0.888 |    0.652 |             0.380 |                 0.619 |  0.600 |
| closed_loop_finite_response_h1_h2_h4_h8 |         2 |       0.593 |       0.905 |    0.492 |             0.080 |                 0.533 |  0.800 |
| closed_loop_finite_response_h1_h2_h4_h8 |         4 |       0.483 |       0.896 |    0.504 |             0.100 |                 0.435 |  0.800 |
| closed_loop_finite_response_h1_h2_h4_h8 |         8 |       0.524 |       0.855 |    0.488 |             0.080 |                 0.471 |  0.600 |
| moving_tangent_interpolated_oracle      |         1 |       0.827 |       0.874 |    0.818 |             0.500 |                 0.827 |  0.900 |
| moving_tangent_interpolated_oracle      |         2 |       0.725 |       0.903 |    0.590 |             0.290 |                 0.725 |  0.600 |
| moving_tangent_interpolated_oracle      |         4 |       0.599 |       0.931 |    0.580 |             0.160 |                 0.599 |  0.778 |
| moving_tangent_interpolated_oracle      |         8 |       0.497 |       0.992 |    0.475 |             0.090 |                 0.497 |  0.600 |
| raw_teacher_label_self_comparison       |         1 |       1.000 |       1.000 |    1.000 |             1.000 |                 1.000 |  1.000 |
| raw_teacher_label_self_comparison       |         2 |       1.000 |       1.000 |    1.000 |             1.000 |                 1.000 |  1.000 |
| raw_teacher_label_self_comparison       |         4 |       1.000 |       1.000 |    1.000 |             1.000 |                 1.000 |  1.000 |
| raw_teacher_label_self_comparison       |         8 |       1.000 |       1.000 |    1.000 |             1.000 |                 1.000 |  1.000 |
| static_local_causal_oracle              |         1 |       0.792 |       0.823 |    0.801 |             0.500 |                 0.792 |  0.800 |
| static_local_causal_oracle              |         2 |       0.708 |       0.901 |    0.625 |             0.330 |                 0.708 |  0.700 |
| static_local_causal_oracle              |         4 |       0.579 |       0.934 |    0.485 |             0.080 |                 0.579 |  0.778 |
| static_local_causal_oracle              |         8 |       0.540 |       0.998 |    0.500 |             0.070 |                 0.540 |  0.800 |

Residual decrease is **not** causal fidelity pass. The frozen finite-response linearity gate failed, so these results are exploratory development only; no independent V15 finalist was eligible. Historical legacy top-k semantic is reported alongside continuous semantic and has not been replaced.

---

## Bundled report 7: `FINITE_RESPONSE_LINEARITY_V15.md`

# Finite-response linearity — V15

The frozen gate requires cosine ≥0.9, relative L2 ≤0.3, and norm ratio 0.7–1.3 for every SNR-qualified tested pair. Tests use one-sided effects relative to the clean state at ε=1: odd symmetry, additivity, homogeneity (factor 2), REC+Conv, and REC+Conv+KV. This is not the trivial algebraic oddness of a central difference definition.

| test        |   qualified |   pass_count |   median_cosine |   median_relative_l2 |   median_norm_ratio |
|:------------|------------:|-------------:|----------------:|---------------------:|--------------------:|
| additive    |          10 |            2 |           0.801 |                0.617 |               0.943 |
| homogeneous |          10 |            3 |           0.865 |                0.493 |               0.783 |
| odd         |          10 |            0 |           0.313 |                1.057 |               1.102 |
| rec_conv    |          10 |            1 |           0.816 |                0.578 |               0.778 |
| rec_conv_kv |          10 |            0 |           0.761 |                0.679 |               0.676 |

`FINITE_RESPONSE_LINEARITY_GATE = False`. The interface is repeatable in the limited retest, but not sufficiently linear under this operating rule. No V15 differential Hessian/secant geometry was promoted from V14. Machine rows: `finite_response_linearity_v15.parquet` and `finite_response_repeat_v15.parquet`.

---

## Bundled report 8: `STRICT_INTERFACE_AUDIT_V15.md`

# V15 strict interface and authorization audit

- V1–V14 results and protocols are read-only parents. V15 base freeze: `8cccf28e4fd09484e1dd2aaf6dcf2abb311695701e439a40eaee6bdee44f7c35`. The original transfer summary was retained; its correction has separate freeze `53fb6908ae059344fad21037d2707d0ce861d6e730118f00857a0d4a6e3205dd`. Cumulative `FINAL_REPORT.md` is the declared mutable exception.
- Diagnostic train / development validation are disjoint: ID SHA256 `b7efb4bab88dd3c2ab6e89ab0b017136e2cc26efe14ced28b8fb0b9867e97544` / `955ea2e3b3e0f05159838a99e88dfd9b752287a425d86a5374500f438f0547c9`. V15 development deliberately reuses the V13-validation/V14-development panel for comparison; it is **not independent confirmation**. No V13 final bank was used, and no V15 finalist evaluation occurred.
- Numerical repeat max relative L2: `0.0`. Frozen finite-response linearity pass: `False`. Development h1 gate: `False`. Finalist: `NO_ELIGIBLE_FINALIST` (digest `ab14457731da214771b5b18e775084c1e23cbf633f15e67cb441b944351ee5d9`). Independent V15 confirmatory bank: `NOT_CREATED_OR_OPENED`.
- h1/h2/h4/h8 development metrics are in the control report; independent h1/h2/h4/h8: **not tested**. Legacy semantic remains separately gated. A continuous-only pass, if any, cannot override the historical full gate.
- H2 remains; `H3_CANDIDATE_INTERFACE_SUPPORTED = FALSE` and H3 complete-state interpretation is unsupported. `ABSOLUTE_REPLACEMENT_NOT_AUTHORIZED`; `AUTONOMOUS_CONTROLLER_AUTHORIZED = FALSE`. V15 control is editing, not absolute persistent-state encoding.
- No new curvature/Hessian claim was made after the finite-response linearity gate failure. Diagnostic FP32 modes changed state storage/consumption semantics and are never counted as canonical model behavior.
- Historical test compatibility: V14's frozen integrity test hashes the entire cumulative `FINAL_REPORT.md`, so the required V15 append makes that one old test fail. Its stored hash and V14 test were **not** changed. V15's parent guard verifies all 1,957 previously tracked paths with only the declared cumulative-report exception; every other historical file matches.

Formal procedural status: **V15-STOP — FINITE_RESPONSE_LINEARITY_GATE_FAILED**. Evidence supports a tested-regime ideal-vs-actual mismatch (V15-B) and the restricted finite-matrix higher-rank pattern (V15-D). Small-scale BF16 quantization is real but is not sufficient to explain every output discrepancy. V15-C/E/F are not established as authorization-level results.

---

## Bundled report 9: `WRITABLE_CAUSAL_RANK_V15.md`

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
