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
