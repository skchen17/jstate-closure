# V37 派生变量依赖审计

Qwen：`(h,Conv)→q,k,v`，`h→G,β,z`，`(S,G,k)→m=kᵀGS`，`(v,m,β)→delta`，`(S,delta,k)→S'→r→norm→mixer`。`delta/TRUE_UPDATE` 依赖旧 S，不是独立于旧状态的新写入。

Falcon：`(h,Conv)→x,B,C`，`h→dt,dA,gate`，`(dt,B,x)→dBx`，`(S,dA,dBx)→S'→r→norm→mixer`。固定 h/Conv 时 dBx 独立于旧 S。缓存写入和输出均由原生重算，不把历史派生张量独立拼接。
