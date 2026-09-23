# V37 原生递推方程与等价校准

Qwen 单 token 实数形式：`S_d=G S; delta=β(v-kᵀ S_d); S'=S_d+k⊗delta; r=qᵀS'`。Falcon：`S'=dA⊙S+dBx; r=CᵀS'+Dx`。实际比较采用安装源码的 dtype/门控/归一化/输出投影顺序，不把代数式当成 BF16 逐位等式。

V37 新校准：Q/F 每模型 `20`/`20` 个状态，各 `3240`/`3240` 个 state×site×cell×stage 等价审计行；最大绝对误差 `0.0`/`0.0`，instrumented logits/cache/endpoints 均 bitwise。V36 旧校准没有替代此步。
