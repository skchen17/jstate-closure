# V36 派生变量依赖审计

## Qwen

`h → (a,b,z)`；`(h,Conv_cache) → (q,k,v)`；`a → g → G`；`b → beta`；`(S,G) → S_d`；`(S_d,k) → m`；`(v,m,beta) → delta`；`(k,delta) → W`；`(S_d,W) → S'`；`(S',q) → r_raw`；`(r_raw,z) → gated_norm → out_proj → mixer → residual`。

因此 `S_d,m,delta,W,S',r_raw` 都依赖旧 REC 状态。V34 的 TRUE_UPDATE `W` 并不是与旧状态独立的新写入；移植 `W_B` 而保留 `S_A` 是可执行的张量干预，却不是依赖一致的“仅写入”反事实。独立于旧状态的当步写入项是 `k⊗beta*v`，状态依赖的移除项是 `-k⊗beta*(kᵀG S)`，必须明确拆开。

## Falcon

`h → (gate,dt,conv_input)`；`(conv_input,Conv_cache) → (x,B,C)`；`dt → dA`；`(dt,B,x) → dBx`；`(S,dA) → old_decay`；`(old_decay,dBx) → S'`；`(S',C,x,D) → r_raw`；`(r_raw,gate) → gated_norm → out_proj → mixer → ssm_out_multiplier → residual`。

固定 h/Conv 时，`dBx` 与旧 `S` 独立，REC 状态差只经 `S*dA` 入读出。任何改变 S 的干预均重算其下游 S'、raw read 与归一化输出；未把历史派生张量伪装成独立变量。两模型角色对应而微观方程不同。
