# V36 原生一 token 递推方程

以实际安装的 Transformers 源码和已冻结的 V34 重放钩子为准：Qwen 文件 SHA-256 `395439341ea5ba4dd14c103e830383caa75686cfaf6b693dd7efb58622224da6`，Falcon 文件 SHA-256 `1a30acb01cc62b253d53230aa50205395ff5aadbcd7411315542a8d385c8253a`。下列是单 token 缓存解码，不外推到预填充/分块路径。

## Qwen3.5-4B

`h:[1,1,2560]` 经 `in_proj_qkv` 与深度可分离 Conv 得 `q,k,v`；查询/键从 16 个 key head 复制到 32 个 value head，按末维 L2 归一化，转置到 `[batch,head,time,dim]`，转 fp32，`q` 再乘 `1/sqrt(128)`。原生缓存 `S:[1,32,128,128]`。`g=-exp(A_log)*softplus(a+dt_bias)`，`beta=sigmoid(b)`，两个控制量来自同一个局部输入 `h`，而 `q,k,v` 同时依赖 Conv 缓存与 `h`。

`G=exp(g)`；`S_d=G*S`；`m=sum_key(S_d*k)`；`delta=beta*(v-m)`；`W=k ⊗ delta`；`S'=S_d+W`；`r_raw=sum_key(S'*q)`。核心读出先转回查询原 dtype（BF16），再逐 value head 用 `RMSNormGated(r_raw,z)`，`z=in_proj_z(h)`，然后拼接 32 heads 并经 `out_proj` 得 mixer output `[1,1,2560]`。线性注意力 decoder layer 把此 mixer output 直接加到 residual，然后经过后层归一化与 MLP。

实数代数下，固定局部输入/Conv 时 `Δr_raw=qᵀ[GΔS-k⊗(beta*kᵀGΔS)]`。执行时还须遵守 BF16/fp32 转换，不能把 BF16 差分再次舍入误称精确线性。

## Falcon-H1-1.5B

`h:[1,1,2048]` 经输入倍率、`in_proj` 与 `mup_vector` 分为 gate、Conv 输入、dt。Conv 缓存滚动后深度可分离卷积与激活产生 `x,B,C`；`dt=clamp(softplus(dt+dt_bias))`，`A=-exp(A_log)`，`dA=exp(dt*A)`，`dBx=dt*B*x`。原生缓存 `S:[1,48,64,256]`，head 48、head_dim 64、group 1、state_dim 256，`B/C` 按 group→head 扩展。`S'=S*dA+dBx`；`r_raw=CᵀS'+D*x`，其中 D 是 direct/skip。随后 `mamba_rms_norm=true`，执行带 gate 的 `norm(r_raw,gate)`，`out_proj` 产生 mixer output `[1,1,2048]`；decoder layer 再乘 `ssm_out_multiplier` 才加入 residual（同时还有 attention 支路）。

固定局部输入/Conv 时，实数代数 `Δr_raw=Cᵀ(dA⊙ΔS)`；实际有限精度用两次原生公式计算后在 fp32 做差。

## 实现等价校准

|模型|校准状态|状态×层×REC 比较|原始读出最大绝对差|原生完整重放|
|---|---|---|---|---|
|Qwen3.5-4B|20|160|0.000000|bitwise 通过|
|Falcon-H1-1.5B|20|160|0.000000|bitwise 通过|

每模型 20 个未用于开发/验证的校准状态，四个预定层，每层 recipient/donor 两种旧状态，共 160 次。原生完整 mixer/缓存/输出重放在 `interface_*.json`；逐原始读出对照在 `equation_audit_*.parquet`。这是实现等价检验，不是未来行为机制证明。
