# V12 complete report

> This is the canonical single-file bundle for this version. It combines the 
> version-specific FINAL_REPORT section, every standalone version report, and 
> an integrity index of the machine-readable records. Standalone reports remain 
> preserved for direct navigation.

## Bundle provenance

- Source commit: `e270629aee6a310491f9b9113056ba720b83c7cc`
- Generated at: `2026-09-16T19:26:13.639415+00:00`
- Included standalone reports: `8`
- Indexed machine-record files: `31`
- Generator: `scripts/build_complete_version_report.py`

## Version summary and adjudication

## V12 — local causal geometry and intrinsic-dimension audit

Formal decision: **V12-A — MORE_DATA_REQUIRED**. Hypothesis remains **H2**. Smallest independently validated writable dimension: **None**. Autonomous controller authorized: **False**.

### Required scientific answers

1. Semantic metric instability materially explains V11: **False**.
2. 512D data/rank saturation: **not identified**; only train sizes `[600]` had adequate centered rank, so `MORE_DATA_REQUIRED`.
3. Local causal effective rank: stable `1.81`, entropy-effective `3.99`, restricted median r90/r95/r99 `4/6/14` within the frozen 64-direction probe.
4. Causal/PCA overlap: see `VARIANCE_VS_CAUSAL_GEOMETRY_V12.md`; conclusions are restricted to the 64-direction operator.
5. Low-variance/high-causal directions: **True**.
6. Tangent rotation is large under the frozen rule: **True**.
7. Locally-low-dimensional/globally-curved state established: **False** unless both low local rank and successful local oracle hold; current outcome does not authorize that claim.
8. Local causal oracle clearly exceeds global PCA: adjudicated from the independent table, but **no method is authorized** unless listed here: `[]`.
9. A 128/256/384/512D local candidate passes h1: **False** (512D local itself was rank-limited).
10. h2/h4/h8 pass: **False** under the all-family frozen gates.
11. Dominant trajectory failure: **direction_rotation_and_semantic_divergence**; finite-horizon ratios were not treated as eigenvalues.
12. Strict full-state replacement succeeded: **False**.
13. Writable dimension is proven higher than predictive dimension: **not proven globally**; current restricted causal spectrum and failed writeback remain consistent with a larger writable state.
14. Strongest supported outcome: **V12-A — MORE_DATA_REQUIRED**.
15. H2 remains: **True**.
16. Upgrade to H3: **False**.
17. Autonomous controller authorization: **False**.

The learned state-dependent decoder was not run because the protocol did not authorize it before causal geometry and strict replacement succeeded.

## Standalone report integrity index

| report | SHA256 |
|---|---|
| `reports/CAUSAL_MEASUREMENT_AUDIT_V12.md` | `31728d5a63a2631f6e11ba85b0d8345333d79408cf26228932f84c936d15d824` |
| `reports/CAUSAL_TANGENT_GEOMETRY_V12.md` | `1ae023fe8e5552ecdda88a6c31f27d6e1509cf92451dd95c398c62988bcf04ae` |
| `reports/CAUSAL_TRAJECTORY_DYNAMICS_V12.md` | `811642d410d7d02505b15da0b62e0cf449bb095034fa06b7482af50aaab594ed` |
| `reports/DATA_RANK_SCALING_V12.md` | `df2f82c1296ee841485debff3fc84efe1404716dd422800ea53429d895d67a63` |
| `reports/LOCAL_CAUSAL_JACOBIAN_V12.md` | `63a4c9d66dcf2dcfdf375649b58e2e0e6164c472f45c50f7249eb1d38be5eba9` |
| `reports/LOCAL_CAUSAL_ORACLE_V12.md` | `2e59a7e100bb4391a92d8aeccf830a5b4340fd6221719033657b9b025b3da5af` |
| `reports/STRICT_STATE_REPLACEMENT_V12.md` | `45b07ee9118ed0530f4154374039c16e93722f66fbfc10979137c5cb243dadb2` |
| `reports/VARIANCE_VS_CAUSAL_GEOMETRY_V12.md` | `a85847f960373f772f78a68f73f2fa6178f41a1c760d7e5b255706f596b51afe` |

## Machine-record integrity index

| record | bytes | SHA256 |
|---|---:|---|
| `results/v12/processed/adjudication_v12.json` | 1049 | `d6f76d7dd312d207656230dd297e226d427397390d63325845d46f754d0e22c1` |
| `results/v12/processed/causal_confirmatory_v12.json` | 461353 | `63d3a7fa1285a0ecf47ef56331a0c329c73b3a53c1c44a00f3cbfa7b01903eb9` |
| `results/v12/processed/causal_confirmatory_v12.parquet` | 146383 | `3a3f788c919a7fcf5e3839fe8c707ebc66acb67c59910d27adcaeda879518f54` |
| `results/v12/processed/causal_tangent_geometry_v12.json` | 2191 | `918c1a2cb6515c4fb607e1fba1d6936dc68ef10a85dc50a2362cc1e2ecea3525` |
| `results/v12/processed/causal_tangent_geometry_v12.parquet` | 9772 | `5475e85d20d7ed1d8407ad474825a2700fef0d57d2c3375dd692cff20a3f681e` |
| `results/v12/processed/causal_trajectory_dynamics_v12.json` | 535 | `5d39220a5c3548636138da7205a2a598a4bd5010c15acca9878e83e34f343552` |
| `results/v12/processed/causal_trajectory_dynamics_v12.parquet` | 9068 | `09000c4bac9da71de8c130bd27ed32d5c1b16ad26e8cc37916b106534217c68e` |
| `results/v12/processed/confirmatory_states_v12.json` | 30532 | `f43aaf3e93a4ad762c7aeefbc6645739359a804f0b2c5a806cf19c9fdac3caab` |
| `results/v12/processed/data_rank_scaling_analysis_v12.json` | 1002 | `3e8a46a94c43de737178ee878a178a5fba6cf6fb0aad4cda59813f26717b2e61` |
| `results/v12/processed/data_rank_scaling_causal_v12.json` | 2548851 | `173683edfb38d3f5ec0862b3c13961987c7c9972b3a253e8b67ed386bc350a89` |
| `results/v12/processed/data_rank_scaling_causal_v12.parquet` | 156347 | `6b5daad40012e8e11bf5c02cc407c4b7d31af6dd7ba53c409ee4636215ce1577` |
| `results/v12/processed/data_rank_scaling_causal_v12_INVALID_hybrid_am3.json` | 858298 | `823fa81e388a75719d66d2280b7bc3886d95ac3c28f1829461a9cbce0e129db5` |
| `results/v12/processed/data_rank_scaling_causal_v12_INVALID_hybrid_am3.parquet` | 28511 | `035ee84f228996daea5ee79419089bf7466f9e62ae673cdd85c174fd45bd91df` |
| `results/v12/processed/data_rank_scaling_v12.parquet` | 9169 | `109b19d9d990ff8ab7ce34d9be148014a729d124f1668c32fa7f14a027fc5ded` |
| `results/v12/processed/development_states_v12.json` | 157146 | `699626efd60e2f3190e0fc645a7afe796465dd40f03ac616507563f3b3560cf1` |
| `results/v12/processed/jvp_probe_directions_v12.json` | 8260 | `b42992fbddeaed8e6cacc5ccff347579a8d12de6bd73f27807fc24cef7b605e4` |
| `results/v12/processed/local_causal_jacobian_v12.json` | 880 | `6b1b0ca6a07dd2a46d57003b260b58ba0963c354061613313e92bfb237c234e5` |
| `results/v12/processed/local_causal_jacobian_v12.parquet` | 20504 | `922f3981e73c0fea0a9e10a4ac207eb1c1690300e202d7085253ef52faaef4d5` |
| `results/v12/processed/local_causal_oracle_stage1_v12.json` | 666726 | `e94fa613284d6bbbd071a35adf84a4d2c27f50bd83b7d6c46c72c0fa8943da39` |
| `results/v12/processed/local_causal_oracle_stage1_v12.parquet` | 194333 | `d03964a1b1d04a7cfe31ae7edd365167bb16f273cc2f825c9836c9f0ea867a3a` |
| `results/v12/processed/local_causal_oracle_stage2_v12.json` | 346095 | `834ab02dc24e6fe3199b4062a92294e35a229a71da02aa6cf4a76fa3de9b1709` |
| `results/v12/processed/local_causal_oracle_stage2_v12.parquet` | 117504 | `30f9bce54db48578dd3ec40798b86fbab6671d2b03019f3c85150a75b701796d` |
| `results/v12/processed/measurement_audit_v12.json` | 3984 | `c73d77b4c3dd402de2b8386bb7f7425bfec4b59fc55b3729e05b987756478f1f` |
| `results/v12/processed/measurement_audit_v12.parquet` | 26395 | `edad149e1c88784ca50aa8c6df8e712f9cf87848c673ad19618bc4f35132d0dd` |
| `results/v12/processed/method_specs_v12.json` | 17441 | `de3245f8a43b331745008e5ee32932258a0a7bdde86ef45bfbdc620a4a7d2af5` |
| `results/v12/processed/report_integrity_v12.json` | 1634 | `054bc6f4d065ebed1051fa75f73f36bd9799198546edbfc8a7fac0a3055ae8fa` |
| `results/v12/processed/scaling_validation_states_v12.json` | 67923 | `d3c4b5bc89f3c8a71c5a5e58f0fc09bb2abab047acf2ed8baeadaa0487882e90` |
| `results/v12/processed/strict_state_replacement_v12.json` | 115940 | `da636be0c9d5016e801be81084d34460976c95f5a264b2e2b95d0b40a99e5128` |
| `results/v12/processed/strict_state_replacement_v12.parquet` | 66077 | `1d0a1f1dcdd743cbbd190ab7015a8949e1c3534524f0d3c6b1bb195fe86b93f0` |
| `results/v12/processed/variance_vs_causal_v12.json` | 803 | `15dbb6d0ff9ac3a0e10028582839e2be82b9cf3909bec261186631cb45ea904d` |
| `results/v12/processed/variance_vs_causal_v12.parquet` | 10987 | `ac1cb9d257292a137d39bc608575c07ccc9ac2cd03a1c39a3cd10660841ea635` |

---

## Bundled report 1: `CAUSAL_MEASUREMENT_AUDIT_V12.md`

# Causal measurement audit — V12

本协议严格区分 predictive fidelity、intervention/writeback fidelity 与 state-replacement fidelity；三者不可互换。冻结的 0.8 semantic gate 未修改。

| condition | continuous cosine | rank corr | top-10 | weighted top-10 | teacher norm |
|---|---:|---:|---:|---:|---:|
| same_family_mix_0.01 | 0.945 | 0.814 | 0.745 | 0.897 | 0.024 |
| scale_0.99 | 0.949 | 0.826 | 0.740 | 0.899 | 0.024 |
| scale_1.01 | 0.951 | 0.829 | 0.795 | 0.906 | 0.024 |

结论：**未发现足以解释 V11 失败的显著 metric instability**。判据为 continuous direction ≥0.95 时 top-k overlap 是否仍跌破 0.8。该审计仅使用 development bank 的 20 个 pair。

Machine record: `results/v12/processed/measurement_audit_v12.parquet` (`edad149e1c88784ca50aa8c6df8e712f9cf87848c673ad19618bc4f35132d0dd`).

---

## Bundled report 2: `CAUSAL_TANGENT_GEOMETRY_V12.md`

# Causal tangent geometry — V12

Principal angles are measured between right-singular subspaces in the same frozen 64-direction coordinate system.

|   grassmann_distance |   maximum_principal_angle_degrees |   mean_principal_angle_degrees |   rank | relation                        |   token_distance |   topk_overlap |
|---------------------:|----------------------------------:|-------------------------------:|-------:|:--------------------------------|-----------------:|---------------:|
|                1.812 |                            84.566 |                         38.544 |      8 | across_family                   |            0.500 |          0.589 |
|                2.793 |                            87.491 |                         43.924 |     16 | across_family                   |            0.500 |          0.511 |
|                3.506 |                            88.240 |                         35.854 |     32 | across_family                   |            0.500 |          0.615 |
|                1.755 |                            84.346 |                         36.533 |      8 | same_prompt_successive_position |            1.000 |          0.614 |
|                2.687 |                            87.277 |                         41.289 |     16 | same_prompt_successive_position |            1.000 |          0.548 |
|                3.431 |                            88.699 |                         34.655 |     32 | same_prompt_successive_position |            1.000 |          0.632 |

- Large tangent rotation (frozen ≥30° rule): `True`.
- Local rank reached the probe boundary: `False`.

`local rank low + angles large` 才支持 curved low-dimensional atlas；若 rank 接近 64-direction probe boundary，则只能报告 probe-limited evidence，不能声称全局低维。

Machine record: `results/v12/processed/causal_tangent_geometry_v12.parquet` (`5475e85d20d7ed1d8407ad474825a2700fef0d57d2c3375dd692cff20a3f681e`).

---

## Bundled report 3: `CAUSAL_TRAJECTORY_DYNAMICS_V12.md`

# Causal trajectory dynamics — V12

- Source split: `confirmatory`
- Error norm monotone: `False`
- trajectory-angle change: `+32.79°`
- semantic-cosine change: `-0.369`
- dominant observed failure mode: **direction_rotation_and_semantic_divergence**

Each record includes `||e_h||`, teacher-trajectory angle, semantic trajectory cosine, and error components parallel/orthogonal to the teacher effect. Finite-horizon ratios are explicitly **not** reported as Jacobian eigenvalues.

Machine record: `results/v12/processed/causal_trajectory_dynamics_v12.parquet` (`09000c4bac9da71de8c130bd27ed32d5c1b16ad26e8cc37916b106534217c68e`).

---

## Bundled report 4: `DATA_RANK_SCALING_V12.md`

# Data × rank scaling — V12

- 可用的 balanced nested train sizes: `[150, 300, 450, 600]`。
- 请求但原始冻结 capture bank 不支持的 sizes: `[1000, 2000, 4000]`，状态为 `NOT_IDENTIFIED_BANK_SIZE`。
- 512D 有充分 empirical rank 的 sizes: `[600]`。
- fixed-512 saturation 是否可识别: `False`。

- `scaling_architecture_joint_pca`: N=150→600 的 d128 direction 变化拟合为 -0.350.
- `scaling_causal_weighted`: N=150→600 的 d128 direction 变化拟合为 +0.328.
- `scaling_combined_pca`: N=150→600 的 d128 direction 变化拟合为 -0.276.

结论：**MORE_DATA_REQUIRED**。现有冻结原始状态只有 N=600 能识别 512D，因此不能声称 512D 已饱和，也不能把 rank 不足 silently clamp 成较小维度。

Offline record SHA256: `109b19d9d990ff8ab7ce34d9be148014a729d124f1668c32fa7f14a027fc5ded`. Causal record SHA256: `6b5daad40012e8e11bf5c02cc407c4b7d31af6dd7ba53c409ee4636215ce1577`.

---

## Bundled report 5: `LOCAL_CAUSAL_JACOBIAN_V12.md`

# Local causal Jacobian / JVP — V12

使用关闭 Flash SDP 后的 **exact autograd JVP**，不是 finite-horizon ratio。输入 operator 限制到冻结的 64 个 empirical joint-PC raw-state directions；target 是 h1/h2/h4 的 256 个 selected-J coordinates 与 16 个 logits。

- sampled local states: `10`
- mean stable rank: `1.81`
- mean effective rank: `3.99`
- median restricted r90/r95/r99: `4` / `6` / `14`

这些数只估计 64-direction empirical restriction 下的 local causal dimension，**不是**完整 raw persistent state 或整个模型的全局 intrinsic dimension。

Machine record: `results/v12/processed/local_causal_jacobian_v12.parquet` (`922f3981e73c0fea0a9e10a4ac207eb1c1690300e202d7085253ef52faaef4d5`).

---

## Bundled report 6: `LOCAL_CAUSAL_ORACLE_V12.md`

# Local causal oracle — V12

Global PCA、global causal covariance basis、current-J-neighborhood local PCA 与 local causal basis 都通过同一 raw-state dual reconstruction interface 比较。Local 512D 因 512-neighbor centered rank≤511 被正式标记 `NOT_IDENTIFIED_RANK_LIMIT`。

## Development h1

| method | h | direction | magnitude | semantic | output | sign | all-family gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| global_architecture_pca_d128 | 1 | 0.206 | 0.337 | 0.062 | 0.252 | 0.520 | FAIL |
| global_architecture_pca_d256 | 1 | 0.431 | 0.388 | 0.170 | 0.420 | 0.640 | FAIL |
| global_architecture_pca_d384 | 1 | 0.870 | 0.851 | 0.534 | 0.833 | 0.740 | FAIL |
| global_architecture_pca_d512 | 1 | 0.923 | 0.968 | 0.620 | 0.897 | 0.860 | FAIL |
| global_architecture_pca_d64 | 1 | 0.189 | 0.326 | 0.066 | 0.225 | 0.620 | FAIL |
| global_causal_basis_d128 | 1 | 0.902 | 0.944 | 0.598 | 0.865 | 0.780 | FAIL |
| global_causal_basis_d256 | 1 | 0.913 | 0.963 | 0.652 | 0.875 | 0.760 | FAIL |
| global_causal_basis_d384 | 1 | 0.918 | 0.960 | 0.620 | 0.886 | 0.780 | FAIL |
| global_causal_basis_d512 | 1 | 0.922 | 0.968 | 0.628 | 0.896 | 0.820 | FAIL |
| global_causal_basis_d64 | 1 | 0.891 | 0.940 | 0.570 | 0.861 | 0.800 | FAIL |
| global_combined_pca_d128 | 1 | 0.188 | 0.336 | 0.084 | 0.228 | 0.520 | FAIL |
| global_combined_pca_d256 | 1 | 0.477 | 0.472 | 0.210 | 0.394 | 0.660 | FAIL |
| global_combined_pca_d384 | 1 | 0.766 | 0.725 | 0.432 | 0.698 | 0.760 | FAIL |
| global_combined_pca_d512 | 1 | 0.924 | 0.977 | 0.614 | 0.894 | 0.800 | FAIL |
| global_combined_pca_d64 | 1 | 0.176 | 0.332 | 0.060 | 0.204 | 0.460 | FAIL |
| local_causal_basis_d128 | 1 | 0.904 | 0.958 | 0.610 | 0.863 | 0.840 | FAIL |
| local_causal_basis_d256 | 1 | 0.915 | 0.974 | 0.618 | 0.885 | 0.760 | FAIL |
| local_causal_basis_d384 | 1 | 0.920 | 0.966 | 0.634 | 0.888 | 0.840 | FAIL |
| local_causal_basis_d64 | 1 | 0.891 | 0.960 | 0.572 | 0.859 | 0.880 | FAIL |
| local_pca_d128 | 1 | 0.253 | 0.382 | 0.112 | 0.252 | 0.560 | FAIL |
| local_pca_d256 | 1 | 0.502 | 0.569 | 0.184 | 0.522 | 0.680 | FAIL |
| local_pca_d384 | 1 | 0.832 | 0.839 | 0.444 | 0.809 | 0.800 | FAIL |
| local_pca_d64 | 1 | 0.207 | 0.350 | 0.082 | 0.255 | 0.580 | FAIL |

## Development staged h2/h4/h8 (h16 only after h1 pass)

| method | h | direction | magnitude | semantic | output | sign | all-family gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| global_architecture_pca_d512 | 2 | 0.816 | 0.970 | 0.350 | 0.778 | 0.760 | FAIL |
| global_architecture_pca_d512 | 4 | 0.681 | 0.981 | 0.228 | 0.640 | 0.674 | FAIL |
| global_architecture_pca_d512 | 8 | 0.521 | 0.999 | 0.104 | 0.508 | 0.720 | FAIL |
| global_causal_basis_d512 | 2 | 0.808 | 0.968 | 0.352 | 0.790 | 0.740 | FAIL |
| global_causal_basis_d512 | 4 | 0.671 | 0.974 | 0.204 | 0.644 | 0.674 | FAIL |
| global_causal_basis_d512 | 8 | 0.532 | 1.009 | 0.104 | 0.518 | 0.580 | FAIL |
| local_causal_basis_d384 | 2 | 0.818 | 0.961 | 0.336 | 0.788 | 0.760 | FAIL |
| local_causal_basis_d384 | 4 | 0.688 | 0.976 | 0.222 | 0.623 | 0.717 | FAIL |
| local_causal_basis_d384 | 8 | 0.516 | 1.002 | 0.096 | 0.488 | 0.640 | FAIL |
| local_pca_d384 | 2 | 0.734 | 0.879 | 0.254 | 0.701 | 0.620 | FAIL |
| local_pca_d384 | 4 | 0.645 | 0.948 | 0.160 | 0.618 | 0.739 | FAIL |
| local_pca_d384 | 8 | 0.520 | 1.010 | 0.088 | 0.522 | 0.680 | FAIL |

## New independent V12 confirmation

| method | h | direction | magnitude | semantic | output | sign | all-family gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| global_architecture_pca_d512 | 1 | 0.918 | 0.958 | 0.644 | 0.896 | 0.840 | FAIL |
| global_architecture_pca_d512 | 2 | 0.806 | 0.972 | 0.342 | 0.772 | 0.820 | FAIL |
| global_architecture_pca_d512 | 4 | 0.677 | 1.000 | 0.192 | 0.638 | 0.600 | FAIL |
| global_architecture_pca_d512 | 8 | 0.531 | 1.001 | 0.106 | 0.519 | 0.776 | FAIL |
| global_causal_basis_d512 | 1 | 0.920 | 0.964 | 0.624 | 0.893 | 0.820 | FAIL |
| global_causal_basis_d512 | 2 | 0.811 | 0.974 | 0.356 | 0.775 | 0.780 | FAIL |
| global_causal_basis_d512 | 4 | 0.663 | 1.003 | 0.210 | 0.618 | 0.689 | FAIL |
| global_causal_basis_d512 | 8 | 0.531 | 1.005 | 0.098 | 0.521 | 0.816 | FAIL |
| local_causal_basis_d384 | 1 | 0.915 | 0.963 | 0.634 | 0.885 | 0.820 | FAIL |
| local_causal_basis_d384 | 2 | 0.808 | 0.978 | 0.340 | 0.777 | 0.800 | FAIL |
| local_causal_basis_d384 | 4 | 0.672 | 1.004 | 0.202 | 0.643 | 0.756 | FAIL |
| local_causal_basis_d384 | 8 | 0.528 | 0.993 | 0.122 | 0.528 | 0.796 | FAIL |
| local_pca_d384 | 1 | 0.829 | 0.846 | 0.486 | 0.818 | 0.800 | FAIL |
| local_pca_d384 | 2 | 0.719 | 0.908 | 0.276 | 0.702 | 0.700 | FAIL |
| local_pca_d384 | 4 | 0.628 | 0.982 | 0.174 | 0.610 | 0.822 | FAIL |
| local_pca_d384 | 8 | 0.515 | 0.996 | 0.094 | 0.502 | 0.755 | FAIL |

- Confirmed authorized methods: `[]`
- Smallest causally validated dimension: `None`
- Formal outcome: **V12-A — MORE_DATA_REQUIRED**

---

## Bundled report 7: `STRICT_STATE_REPLACEMENT_V12.md`

# Strict state replacement — V12

- Tested scope: `formal audit of absolute full persistent/model-cache replacement; writeback diagnostic retained separately`
- Raw target-state bypass detected: `False`
- Compact-only full-state verified: `False`
- Strict full-state replacement pass: `False`
- Reason: the frozen V12 representations encode persistent intervention deltas; they do not encode an absolute complete REC/conv/KV/model-cache state.

The diagnostic overwrites/reconstructs the intervention-bearing persistent channels through the strict prepared-state interface. It does **not** relabel this as complete model-cache replacement: unmodeled cache fields remain a structural scaffold. Consequently it cannot authorize H3 or an autonomous controller.

| method | h | direction | magnitude | semantic | output | sign | all-family gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| local_causal_basis_d384 | 1 | 0.915 | 0.963 | 0.634 | 0.885 | 0.820 | FAIL |
| local_causal_basis_d384 | 2 | 0.808 | 0.978 | 0.340 | 0.777 | 0.800 | FAIL |
| local_causal_basis_d384 | 4 | 0.672 | 1.004 | 0.202 | 0.643 | 0.756 | FAIL |
| local_causal_basis_d384 | 8 | 0.528 | 0.993 | 0.122 | 0.528 | 0.796 | FAIL |

Machine record: `results/v12/processed/strict_state_replacement_v12.parquet` (`1d0a1f1dcdd743cbbd190ab7015a8949e1c3534524f0d3c6b1bb195fe86b93f0`).

---

## Bundled report 8: `VARIANCE_VS_CAUSAL_GEOMETRY_V12.md`

# Variance directions vs causal directions — V12

At rank 16 in the frozen 64-direction operator:

- PCA/causal subspace overlap: `0.145`
- mean principal angle: `71.92°`
- causal sensitivity captured by leading PCA directions: `0.023`
- low-variance/high-causal-sensitivity directions present: `True`
- REC/conv/KV contribution: `0.347` / `0.379` / `0.274`

This comparison is restricted to the frozen empirical probe domain; it does not identify the full raw-state spectrum.

Machine record: `results/v12/processed/variance_vs_causal_v12.parquet` (`ac1cb9d257292a137d39bc608575c07ccc9ac2cd03a1c39a3cd10660841ea635`).
