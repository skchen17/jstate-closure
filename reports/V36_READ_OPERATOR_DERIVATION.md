# V36 原生 REC 读算子推导

在固定 h 和 Conv 后，Qwen 的 `M_Q(u)` 是 `qᵀ[G· - k⊗(beta*kᵀG·)]`；Falcon 的 `M_F(u)` 是 `Cᵀ(dA⊙·)`。这里 u 含实际 post-Conv q/k/v 或 x/B/C 与来自 h 的门控/衰减；不是仅凭层名推测读写。当前输入驱动项（Qwen `k⊗beta*v`、Falcon `dBx`）在 REC 差分中消去，Falcon direct/skip `D*x` 也消去。归一化/门控与输出投影是读后另一个非线性阶段，不能简单并入线性 M。

预测在原始读出处用同一源代码顺序、dtype 转换计算 `raw(S_B,u)-raw(S_A,u)`，先分别 BF16→fp32，再相减。将两个 BF16 读出先相减会额外舍入，曾在无正式记录的调试运行中被检测并修正；正式记录采用前者。源代码 SHA 与逐状态等价校准见原生方程报告。
