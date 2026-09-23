# V37 状态读算子推导

固定 h/Conv，Qwen 的实数线性状态读映射为 `qᵀ[G I - βk(kᵀG)]S`；Falcon 为 `Cᵀ(dA⊙S)`。两者均是局部状态差对 raw read 的映射。V37 用真实有限精度方程分别算 3×3 预测，再与原生运行比较；归一化/门控和 output projection 另作后读阶段，不直接等同 raw 线性算子。
