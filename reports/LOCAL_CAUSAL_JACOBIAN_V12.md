# Local causal Jacobian / JVP — V12

使用关闭 Flash SDP 后的 **exact autograd JVP**，不是 finite-horizon ratio。输入 operator 限制到冻结的 64 个 empirical joint-PC raw-state directions；target 是 h1/h2/h4 的 256 个 selected-J coordinates 与 16 个 logits。

- sampled local states: `10`
- mean stable rank: `1.81`
- mean effective rank: `3.99`
- median restricted r90/r95/r99: `4` / `6` / `14`

这些数只估计 64-direction empirical restriction 下的 local causal dimension，**不是**完整 raw persistent state 或整个模型的全局 intrinsic dimension。

Machine record: `results/v12/processed/local_causal_jacobian_v12.parquet` (`922f3981e73c0fea0a9e10a4ac207eb1c1690300e202d7085253ef52faaef4d5`).
