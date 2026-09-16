# Causal measurement audit — V12

本协议严格区分 predictive fidelity、intervention/writeback fidelity 与 state-replacement fidelity；三者不可互换。冻结的 0.8 semantic gate 未修改。

| condition | continuous cosine | rank corr | top-10 | weighted top-10 | teacher norm |
|---|---:|---:|---:|---:|---:|
| same_family_mix_0.01 | 0.945 | 0.814 | 0.745 | 0.897 | 0.024 |
| scale_0.99 | 0.949 | 0.826 | 0.740 | 0.899 | 0.024 |
| scale_1.01 | 0.951 | 0.829 | 0.795 | 0.906 | 0.024 |

结论：**未发现足以解释 V11 失败的显著 metric instability**。判据为 continuous direction ≥0.95 时 top-k overlap 是否仍跌破 0.8。该审计仅使用 development bank 的 20 个 pair。

Machine record: `results/v12/processed/measurement_audit_v12.parquet` (`edad149e1c88784ca50aa8c6df8e712f9cf87848c673ad19618bc4f35132d0dd`).
