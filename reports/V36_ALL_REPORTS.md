# V36 All Reports — Computational Origin of REC–Conv Conditional Effects

This is the standalone single-file bundle for V36; topic reports remain separately preserved.

## Topic report hashes

|Report|SHA-256|
|---|---|
|V36_COMPLETE_REPORT.md|99e77c556c675c5dcdbd266f70c6347cfbdefce688ef306b902ea7a223058179|
|V36_COUNTERFACTUAL_OPERATOR_SWAP.md|24cc2902d71153c9ad252a8d69f12b7b3e836ba08a5b0842d7b74b9c11dc4d28|
|V36_CROSS_MODEL_OPERATOR_LAW.md|8e03c288a7aad29c5b6255b05d3f48ee59d4943f980dcde5a88c85360cad0907|
|V36_DERIVED_VARIABLE_DEPENDENCY_AUDIT.md|fd01c4d096ad2099ea0ddb3f02aa2045de83ef5476fadf88b223cffb6d0232cb|
|V36_EXECUTION_MANIFEST.md|1bed74394de58eafaaf02352326f846672e1385eb5bb58cf6791070cf6cbc8d5|
|V36_HIGH_LEVEL_RECONFIRMATION.md|5527f7860cf159bd1c971b33f053e38ac1f994880837dc6532edce67d69f4370|
|V36_LOCAL_FIXED_INPUT_FACTORIAL.md|b0c54fe28b4beca35debcd6a655109c532b867fd7fd24f4065aff0d066d452f5|
|V36_LOCAL_TO_GLOBAL_REINSERTION.md|9800deff43d40bc370f399330cbb95beeaab107db288905ab71958548425dd6a|
|V36_MATCHED_OPERATOR_PREDICTION.md|cdfc206f23b1b9592670c06651c78f012640ef431dc2732fc467de9f6d6d8c2d|
|V36_NATIVE_OPERATOR_COMPOSITION.md|bd243cb11973f0ccd175eb5d696b1428f11afa95ec64f9801eb9ae93c14bfeb7|
|V36_NATIVE_RECURRENCE_EQUATIONS.md|5020dcc426cc029ec5136989bb09359b1097295a66e1c12d7d06a29d2e8f7735|
|V36_OLD_STATE_VS_UPDATE.md|4cdb37cb83b529669167871ec51bb507a93847bbd399f4cac04b28b7210ba375|
|V36_PRE_POST_NORMALIZATION.md|80459a1d9f4080d677cd03e3a6cfc5f61eb6fe0b0fa5209d14d0c5982d87845d|
|V36_READ_OPERATOR_DERIVATION.md|3aee1e5f1b98398a8d2e114b53763d6718d0766fbb47705a2bd427ba0d01f24a|
|V36_SCIENTIFIC_ANSWERS.md|a627443955f816e3058e35a16fe32271c1c058716a6f3e005c660f45c4c65269|
|V36_STRICT_INTERFACE_AUDIT.md|3d2875fcab3b6736df1b44a0a18ed369110200b516d1cad9de6e3970e09df44d|
|V36_TASK_GROUNDED_FUNCTIONAL_TEST.md|ba7a944f54c608e8e734a8bfbce745dc7d6404bf0e070b65cf6d90d47cd6a342|
|V36_THREE_WAY_COMPATIBILITY_MATRIX.md|744d8174723e4c797e87f07533d1da6a3fac96fc19dd2536231e9236017e452d|
|V36_TRUE_UPDATE_REINTERPRETATION.md|1fded65d8e59f14de6f2aa0338e5bd2b2c7ccd3bc58704869b74a8f8cfc9e609|
|V36_UPSTREAM_CONTEXT_DEPENDENCE.md|b0ec9a97a4b7f6c79a31ad4dd6ecaaa9c6953e4775c0bdf985aa7e0516f62453|
|V36_V35_ADJUDICATION_AUDIT.md|89340b40c14858cf605197cc6aa84191df1141efa2885b1309349d334c081ead|

---

## V36_COMPLETE_REPORT.md

# V36 Complete Report — Computational Origin of REC–Conv Conditional Effects

**Does Conv Change How Persistent Recurrent State Is Read?**

结论：原生递推式对两架构固定输入的 REC 状态差读出预测精确，但所测四个预定层既没有稳健的供体 Conv 匹配优势，也无法单层重现大部分全模型条件效应。V36 的强计算机制假说未通过开发/验证联合门槛；独立最终保持封存。不能把局部代数恒等式当成跨模型行为机制。

## V35 审计（与 V36 结局分离）

`V35_AUDIT_ADJUDICATION_AMENDMENT_REQUIRED`：V35 Q2+Q3+Q4 开发/验证两模型强门槛通过，但候选生成遗漏三段组合；已有 FULL_DEPTH_ONLY 最终不检验它。修订只追加，历史文件不覆盖。

## 新面板高层现象

|角色|模型|状态|正收益率|中位修正减少|残差对齐|家族|门槛|
|---|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|80|1.000|0.335|0.747|5|通过|
|development|Falcon-H1-1.5B|80|1.000|0.386|0.790|5|通过|
|validation|Qwen3.5-4B|40|1.000|0.412|0.815|5|通过|
|validation|Falcon-H1-1.5B|40|1.000|0.455|0.841|5|通过|

## 局部机制主判定

|角色|模型|状态|局部公式余弦|幅度排序|匹配优势率|下游恢复率|联合门槛|
|---|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|80|1.000|1.000|0.250|0.030|未通过|
|development|Falcon-H1-1.5B|80|1.000|1.000|0.500|0.020|未通过|
|validation|Qwen3.5-4B|40|1.000|1.000|0.250|0.032|未通过|
|validation|Falcon-H1-1.5B|40|1.000|1.000|0.417|0.028|未通过|

预定匹配优势 ≥0.80、单层下游恢复 ≥0.20、至少 4/5 家族；两模型两角色均失败。固定/自然交互范数比在开发 Q/F 为 `0.185`/`0.117`，验证为 `0.187`/`0.109`。这支持强上游依赖的描述性判断，但未建立新的全深度计算机制。

## 原生代数、对照和解释边界

两模型各 20 校准状态、160 原始读出对照零误差；开发/验证预测余弦与幅度排序均达到 1.0。Qwen TRUE_UPDATE 旧状态依赖，Falcon dBx 固定输入下旧状态独立；读后归一化在两架构尺度相反。供体 mixer 拷贝为接口上界，不是机制。全部对照与 hash 在专题报告/机器索引。

## 封存与后继

最终开放记录 `opened=false`，40+40 独立最终响应未观察。跨层自然组合、未来 3×3、外部语义任务按协议不授权。不能确认 V36-A/B/D/E/F/G/H；C 为描述性上游依赖，I 只限“已测试局部候选不足”，不得泛化为不存在其他计算律。下一阶段需冻结新的跨层/上游上下文假说与新样本，保留 V34/V35 全读出因果切面，不称显式误差纠正。

---

## V36_COUNTERFACTUAL_OPERATOR_SWAP.md

# V36 状态×算子反事实交换

已执行固定 h 下 A/B/C 的局部状态×Conv 天然缓存互换，并验证请求/实现哈希及原生重放。`STATE_B+OPERATOR_A` 与 `STATE_A+OPERATOR_B` 不能仅凭张量不同判定未来行为归属。因局部匹配/下游门槛失败，作为正式候选的整模型未来交换与完整九格未来兼容矩阵未开启；不存在可报告的“匹配配对恢复供体未来”结论。

---

## V36_CROSS_MODEL_OPERATOR_LAW.md

# V36 跨模型读算子法则

Qwen 与 Falcon 各自的原生公式均对固定局部输入下的旧 REC 状态差预测精确；这是架构内计算等价，不是两模型共享的行为机制确认。

|角色|模型|状态|局部公式余弦|幅度排序|匹配优势率|下游恢复率|联合门槛|
|---|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|80|1.000|1.000|0.250|0.030|未通过|
|development|Falcon-H1-1.5B|80|1.000|1.000|0.500|0.020|未通过|
|validation|Qwen3.5-4B|40|1.000|1.000|0.250|0.032|未通过|
|validation|Falcon-H1-1.5B|40|1.000|1.000|0.417|0.028|未通过|

两模型开发/验证均未达到匹配优势和下游插入门槛，且读后归一化尺度方向不同。因此 V36-A 与 H 不成立为已确认结论；不能推到第三模型或应用。

---

## V36_DERIVED_VARIABLE_DEPENDENCY_AUDIT.md

# V36 派生变量依赖审计

## Qwen

`h → (a,b,z)`；`(h,Conv_cache) → (q,k,v)`；`a → g → G`；`b → beta`；`(S,G) → S_d`；`(S_d,k) → m`；`(v,m,beta) → delta`；`(k,delta) → W`；`(S_d,W) → S'`；`(S',q) → r_raw`；`(r_raw,z) → gated_norm → out_proj → mixer → residual`。

因此 `S_d,m,delta,W,S',r_raw` 都依赖旧 REC 状态。V34 的 TRUE_UPDATE `W` 并不是与旧状态独立的新写入；移植 `W_B` 而保留 `S_A` 是可执行的张量干预，却不是依赖一致的“仅写入”反事实。独立于旧状态的当步写入项是 `k⊗beta*v`，状态依赖的移除项是 `-k⊗beta*(kᵀG S)`，必须明确拆开。

## Falcon

`h → (gate,dt,conv_input)`；`(conv_input,Conv_cache) → (x,B,C)`；`dt → dA`；`(dt,B,x) → dBx`；`(S,dA) → old_decay`；`(old_decay,dBx) → S'`；`(S',C,x,D) → r_raw`；`(r_raw,gate) → gated_norm → out_proj → mixer → ssm_out_multiplier → residual`。

固定 h/Conv 时，`dBx` 与旧 `S` 独立，REC 状态差只经 `S*dA` 入读出。任何改变 S 的干预均重算其下游 S'、raw read 与归一化输出；未把历史派生张量伪装成独立变量。两模型角色对应而微观方程不同。

---

## V36_EXECUTION_MANIFEST.md

# V36 执行清单

父提交 `9c272d501f6c0beb6ee0bf6c643dbdc77aef9475`；基础协议 SHA-256 `19e4af2367bdfcbde1a1f5b9f6dc066818f48462725241002985e0d0aed8a8ac`。语义配对且排除 V28–V35 的样本：校准 `20`、开发 `80`、验证 `40`、封存独立最终 `40`，两模型各同样数量。状态分组哈希：`{"calibration": "a019cb834bb72a570678d91175c663bbb39fcccc747dbd5452d814c783ee64df", "development": "6b48236b681f96fb77e710664ca0c6b71ab42f13df3fd6e549e45882ebe25f15", "independent_final": "d87428e03db6941672be4af81e92f57ecb39ceca71b135147b2c0e986e64edb2", "validation": "b2a97e6855d2fd4dd89cb3d4139d7e6f27189fc915835ad75b52ab32d1521ed3"}`。实际 Qwen/Falcon 层位在局部因子报告；token triple、probe、REC/Conv、原生因子与插入请求/实现摘要在机器记录与索引中。

V35 审计结局 `V35_AUDIT_ADJUDICATION_AMENDMENT_REQUIRED`；开发/验证高层重确认均通过；局部候选两角色均未通过；V36 限定范围结局 `['V36-I_NO_SIMPLE_COMPUTATIONAL_LAW_IDENTIFIED']`；最终开放记录 `opened=false`，40+40 最终状态从未产生响应。跨层组合、语义任务测试依冻结规则不执行。没有调门槛、没有修改历史 V1–V35 记录。

---

## V36_HIGH_LEVEL_RECONFIRMATION.md

# V36 新面板高层重确认

|角色|模型|状态|正收益率|中位修正减少|残差对齐|家族|门槛|
|---|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|80|1.000|0.335|0.747|5|通过|
|development|Falcon-H1-1.5B|80|1.000|0.386|0.790|5|通过|
|validation|Qwen3.5-4B|40|1.000|0.412|0.815|5|通过|
|validation|Falcon-H1-1.5B|40|1.000|0.455|0.841|5|通过|

同一状态的六个 probe 先合并为一个五块标准化向量，再用状态为独立 bootstrap 单位。表中正收益、减少、对齐和家族门槛均来自 `high_level_*_v36.json` 与 factorial Parquet。REC-only 弱于 joint、Conv-only 已有较大供体方向贡献；这里只可称条件 REC 效应，不称模型内部显式误差向量。两模型开发/验证都获准进入局部机制实验；独立最终仍受另外的机制门槛约束。

---

## V36_LOCAL_FIXED_INPUT_FACTORIAL.md

# V36 同输入局部 REC×Conv 因子实验

每个模型仅用冻结的相对层位 2/8/14/20（Qwen 实际层 `[2, 10, 18, 26]`；Falcon `[2, 8, 14, 20]`）。在 Conv-only 自然轨迹捕获同一个 mixer 输入 h；局部构造 `S_A,S_B,S_C × Conv_A,Conv_B,Conv_C` 九格，主要四格为 AA/BA/AB/BB。每个格子以原生 mixer/缓存重算，而不是直接复制 donor 输出。原始、归一化、mixer 与自然轨迹交互均保存于逐 probe Parquet；局部输入、REC/Conv 请求与实现哈希保存于 proof Parquet。

|角色|模型|状态|局部公式余弦|幅度排序|匹配优势率|下游恢复率|联合门槛|
|---|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|80|1.000|1.000|0.250|0.030|未通过|
|development|Falcon-H1-1.5B|80|1.000|1.000|0.500|0.020|未通过|
|validation|Qwen3.5-4B|40|1.000|1.000|0.250|0.032|未通过|
|validation|Falcon-H1-1.5B|40|1.000|1.000|0.417|0.028|未通过|

三条件方向/幅度公式预测为真，但这个代数等价本身不说明供体匹配或下游充分性。四层没有按验证结果适应性重选。

---

## V36_LOCAL_TO_GLOBAL_REINSERTION.md

# V36 局部结果的整模型重插入

在 Conv-only 全模型支路的单一预定层，用同输入重算的 `BB` mixer output 或单独的交互增量替换该层输出，然后让剩余层自然传播到同一六 probe 终点。未复制任何下游 donor 输出。每状态四层结果在统计前聚合；自然 REC 收益低于冻结绝对分母 1.0 的状态不计算不稳定比例。

|角色|模型|状态|局部公式余弦|幅度排序|匹配优势率|下游恢复率|联合门槛|
|---|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|80|1.000|1.000|0.250|0.030|未通过|
|development|Falcon-H1-1.5B|80|1.000|1.000|0.500|0.020|未通过|
|validation|Qwen3.5-4B|40|1.000|1.000|0.250|0.032|未通过|
|validation|Falcon-H1-1.5B|40|1.000|1.000|0.417|0.028|未通过|

冻结要求状态中位局部恢复率 ≥0.20 且 ≥4/5 家族；开发与验证两模型均未通过。单层局部作用存在，却不足以重建整体 REC 条件收益。供体 mixer 输出复制只作为上界对照，见严格接口审计。

---

## V36_MATCHED_OPERATOR_PREDICTION.md

# V36 供体状态与 Conv 读算子匹配预测

固定 `ΔS=S_B-S_A`，在自然 A、匹配 B、第三自然 token C 的 Conv 条件下，先用原生方程预测原始读出差，再运行本地 mixer。状态级中位数如下；六 probe×四层不作为独立样本。

|角色|模型|状态|局部公式余弦|幅度排序|匹配优势率|下游恢复率|联合门槛|
|---|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|80|1.000|1.000|0.250|0.030|未通过|
|development|Falcon-H1-1.5B|80|1.000|1.000|0.500|0.020|未通过|
|validation|Qwen3.5-4B|40|1.000|1.000|0.250|0.032|未通过|
|validation|Falcon-H1-1.5B|40|1.000|1.000|0.417|0.028|未通过|

方向余弦和三条件幅度排序达到 1.0，说明源代码公式被忠实复现；但“B 的状态效应大于 A 与 C”的预定匹配优势率远低于 0.80，且四层重插入 <0.20。五家族没有一类达到联合门槛。不能把精确代数等式误当成匹配假说被证实。

---

## V36_NATIVE_OPERATOR_COMPOSITION.md

# V36 原生算子跨层组合

未执行。冻结规则要求局部算子律在两模型开发和验证通过预测、匹配优势、下游插入与家族联合门槛后才开启跨 24 recurrent 层的自然组合。两模型均未达到这些门槛；因此不把历史全深度 read-output replacement 重新包装成原生算子组合，也不复制别的轨迹的下游输出。

---

## V36_NATIVE_RECURRENCE_EQUATIONS.md

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

---

## V36_OLD_STATE_VS_UPDATE.md

# V36 旧状态与当步更新

Qwen：状态差同时改变 `G*S` 与经 `kᵀG*S` 计算的移除/TRUE_UPDATE；不应把 TRUE_UPDATE 的差称为独立新写入。Falcon：固定 h/Conv 的 `dBx` 与 S 独立，REC 差在原始读出只经旧状态衰减支路。下表是基于原生项计算的原始读出投影及供体 mixer 输出上界，并非每项独立下游因果充分性证明。

|角色|模型|供体混合器上界恢复率|旧状态投影|状态依赖更新投影|打乱算子余弦|反号对向余弦|
|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|0.039|1.205|-0.204|0.988|0.999|
|development|Falcon-H1-1.5B|0.064|0.998|0.000|0.996|0.998|
|validation|Qwen3.5-4B|0.037|1.210|-0.211|0.987|0.999|
|validation|Falcon-H1-1.5B|0.076|0.998|0.000|0.996|0.998|

OLD_ONLY、UPDATE_ONLY 的严格依赖一致构造在固定 h/Conv 下分别退化为完整 REC 差与零 REC 差；若用 donor 状态派生的更新搭配 recipient 旧状态，那是混合反事实，须单独标为干预，不能当纯写入。V36-D/E/F 不凭张量范数正式确认。

---

## V36_PRE_POST_NORMALIZATION.md

# V36 读出归一化前后

|角色|模型|局部 raw→norm 交互幅度中位比|旧状态投影|状态依赖更新投影|
|---|---|---|---|---|
|development|Qwen3.5-4B|149.676|1.205|-0.204|
|development|Falcon-H1-1.5B|0.091|0.998|0.000|
|validation|Qwen3.5-4B|154.395|1.210|-0.211|
|validation|Falcon-H1-1.5B|0.095|0.998|0.000|

逐 probe 的 `I_raw`、归一化交互、mixer 交互范数及 raw→norm 余弦保存在 `local_*.parquet`。Qwen 的大幅度比与 Falcon 的小幅度比表明读后变换的尺度效应显著不同；该比例在 raw 很小时会很大，不能单独证明“非线性创造了大部分全模型纠正”。在两模型均失败的下游恢复门槛下，V36-G 不确认。

---

## V36_READ_OPERATOR_DERIVATION.md

# V36 原生 REC 读算子推导

在固定 h 和 Conv 后，Qwen 的 `M_Q(u)` 是 `qᵀ[G· - k⊗(beta*kᵀG·)]`；Falcon 的 `M_F(u)` 是 `Cᵀ(dA⊙·)`。这里 u 含实际 post-Conv q/k/v 或 x/B/C 与来自 h 的门控/衰减；不是仅凭层名推测读写。当前输入驱动项（Qwen `k⊗beta*v`、Falcon `dBx`）在 REC 差分中消去，Falcon direct/skip `D*x` 也消去。归一化/门控与输出投影是读后另一个非线性阶段，不能简单并入线性 M。

预测在原始读出处用同一源代码顺序、dtype 转换计算 `raw(S_B,u)-raw(S_A,u)`，先分别 BF16→fp32，再相减。将两个 BF16 读出先相减会额外舍入，曾在无正式记录的调试运行中被检测并修正；正式记录采用前者。源代码 SHA 与逐状态等价校准见原生方程报告。

---

## V36_SCIENTIFIC_ANSWERS.md

# V36 35 个科学问题的审慎答案

1–2. V35 Q2+Q3+Q4 四格均通过，但候选构造只收集单段/两段，遗漏已列入 priority 的三段；需追加判定/报告修订，不能追认独立最终。
3–4. Qwen `S'=GS+k⊗beta(v-kᵀGS), r=qᵀS'`；Falcon `S'=dA⊙S+dBx, r=CᵀS'+Dx`，详见原生方程报告。
5–6. Qwen 的 memory、delta、TRUE_UPDATE 依赖旧 S，q/k/v 依赖 Conv；Falcon dBx 不依赖 S，B/C/x 依赖 Conv，dA 来自 h。
7–10. 同输入局部交互存在但相对自然交互很小，单层下游重插入恢复率远低于 0.20；固定 h 后交互衰减，提示上游上下文重要，但不能宣称全部由上游造成。
11–14. 原生公式对未见开发/验证状态、token、五家族的局部 raw 差方向/排序精确；这是源代码等价，非未来行为预测充分性。
15–17. donor Conv 并非稳定最强匹配；第三自然 token 不支持预定 ≥80% 优势。局部九格曾执行但未保存完整矩阵；未来九格未授权，不能宣称对角兼容。
18–20. 固定输入下 Falcon REC 差由旧状态路径进入，Qwen 旧状态也改变状态依赖移除；范数/投影不能单独确认全局 D/E/F。
21. 归一化尺度效应两模型不同；无跨模型充分下游中介证据，不确认 G。
22–23. Qwen TRUE_UPDATE 依赖旧 S，单独移植漏掉/错配旧状态；Falcon dBx 固定输入与旧 S 独立。依赖一致重算解释为什么单独替换不等于完整条件效应，不把历史失败解读为更新毫无作用。
24–28. 两模型局部代数都成立，但匹配优越/下游充分性失败；不能宣称跨模型读算子行为律或预先预测未来 donor 纠正。
29. 外部定义的任务变量测试未获授权，未测。
30–34. V36-A/B/H 未确认；独立最终封存。V36-C 与 D–G 缺少预定的充分因果门槛，不能正式确认；V36-I=true 仅作所测四层局部解释不足的有界结论。
35. 下一模型/应用不能冻结“匹配 read operator”或单层局部机制；可保留已证实的原生局部方程与 V34/V35 全读出切面，下一步需预先设计上游输入/跨层交互实验，且重新冻结新独立样本。

---

## V36_STRICT_INTERFACE_AUDIT.md

# V36 严格接口与对照审计

控制 A：校准完整一 token 原生重放 bitwise；每模型 20 状态、四层、两 REC 状态的公式 raw read 零误差。B/C/I：正式 factorial 的 recipient REC、donor REC × recipient/donor Conv 与自然 joint。D/E：固定 h 的第三自然 token Conv 与 REC 缓存九格，写回哈希在 proof 记录。F：Qwen q/k、Falcon C 的打乱算子组件，仅作为非天然算子诊断。G：合成反号 ΔS 的代数检查，不冒称自然 token。H：自然供体 mixer output 放入 Conv-only 支路作为单层上界，不冒称机制。所有运行均保持 recipient-native KV 与同一六 probe。F/G 不在自然状态流形，不能拿它们的输出当因果任务结果。

|角色|模型|供体混合器上界恢复率|旧状态投影|状态依赖更新投影|打乱算子余弦|反号对向余弦|
|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|0.039|1.205|-0.204|0.988|0.999|
|development|Falcon-H1-1.5B|0.064|0.998|0.000|0.996|0.998|
|validation|Qwen3.5-4B|0.037|1.210|-0.211|0.987|0.999|
|validation|Falcon-H1-1.5B|0.076|0.998|0.000|0.996|0.998|

请求/实现 REC、Conv、mixer 哈希、local h 哈希、完整自然支路响应哈希和独立最终封存记录分别见 `local_proof_*`、`controls_proof_*`、factorial audit 与 `final_opening_v36.json`。未把 activation similarity 作为机制判据。

---

## V36_TASK_GROUNDED_FUNCTIONAL_TEST.md

# V36 任务语义功能测试

未执行。协议只在两模型同一计算机制经开发和验证通过后允许用外部定义的布尔值、绑定、状态转换和模值，测正确答案 logit margin、分类准确率及 donor/recipient 响应忠实度。本版无合格机制候选，独立最终也未开启；不能由五个任务家族的向量响应直接推断语义变量恢复。

---

## V36_THREE_WAY_COMPATIBILITY_MATRIX.md

# V36 三自然 token 的 3×3 兼容性矩阵

局部固定输入阶段确实逐 probe 运行了 `S_i × Conv_j` 九格（i,j=A/B/C），每格请求/实现的 REC、Conv 哈希在 `local_proof_*.parquet`。其中 A/B 的原始读出、归一化读出与 mixer 交互进入主记录；C 格用于冻结的第三 token 算子预测。各九格完整输出矩阵没有独立落盘，故**不报告**任何未保存的逐格数值或“对角更优”结论。

由于匹配与局部下游门槛均未过，未来响应的九格矩阵按预先规则未获授权。后续若要检验对角兼容性，须新实验先冻结非循环局部/未来评分，而不能从现有 donor 误差直接定义匹配再用同一误差评估。

---

## V36_TRUE_UPDATE_REINTERPRETATION.md

# V34 TRUE_UPDATE 失败的 V36 重新解释

Qwen 的 `W=k⊗beta*(v-kᵀG S)` 依赖旧 S。只替换 W 而不替换 `G S`，既漏掉旧状态项，也让 donor 派生的移除项与 recipient 旧状态不一致；所以 V34 TRUE_UPDATE 单独失败不能证明更新无关。Falcon 的 `dBx=dt*B*x` 固定输入时与 S 无关，TRUE_UPDATE 交换没有携带 REC 差；REC 的条件作用通过 `S*dA` 被 C 读取。两种架构原因不同。

本版在校准与开发/验证的公式重算支持上述依赖关系；没有对 V34 全深度 TRUE_UPDATE 历史数据作追认或覆写，也未声称重跑了一个全深度更新干预。

---

## V36_UPSTREAM_CONTEXT_DEPENDENCE.md

# V36 上游上下文依赖

NATURAL_FORWARD 让 AA/BA/AB/BB 各有自然上游输入；FIXED_LOCAL_INPUT 将四格都置于 Conv-only 的同一个 mixer 输入。记录两种交互的逐 probe 范数与余弦，并以状态中位为单位。

|角色|模型|固定/自然交互范数中位比|局部下游收益恢复率|
|---|---|---|---|
|development|Qwen3.5-4B|0.185|0.030|
|development|Falcon-H1-1.5B|0.117|0.020|
|validation|Qwen3.5-4B|0.187|0.032|
|validation|Falcon-H1-1.5B|0.109|0.028|

固定输入交互显著缩小，是上游变化很重要的证据；但未在协议中冻结 V36-C 的单独充分门槛，因此只作描述性支持，不正式宣称上游路径单独解释了全部条件收益。

---

## V36_V35_ADJUDICATION_AUDIT.md

# V36 pre-experiment audit of V35 adjudication

Status: `V35_AUDIT_ADJUDICATION_AMENDMENT_REQUIRED` (with a reporting correction). This is an append-only interpretation amendment, not a replacement for the sealed V35 decision or an opening of its independent final.

## Verified evidence

The frozen V35 configuration predeclared `Q2_Q3_Q4` in `subset_conditions` and `late_cumulative`, and `development_plan_v35.py` included `LOCALIZED_Q2_Q3_Q4` in `FINAL_PRIORITY`. The sealed depth records show that the triple passed the same `quartile_gate` in both models and both prospective roles:

| Role | Model | REM | REST | cosine | passing families |
|---|---|---:|---:|---:|---:|
| Development | Qwen | 0.990 | 0.824 | 0.914 | 5/5 |
| Development | Falcon | 0.920 | 0.918 | 0.945 | 5/5 |
| Validation | Qwen | 1.013 | 0.825 | 0.901 | 5/5 |
| Validation | Falcon | 0.883 | 0.916 | 0.942 | 5/5 |

The discrepancy is not a family-level failure or a separate frozen threshold. `depth_analysis_v35.py:144` builds `strong_pair_passes` **only** from `pair_conditions`; `Q2_Q3_Q4` is a triple, so it never enters the shared candidate aggregation at line 239. `development_plan_v35.py:candidates()` then creates `LOCALIZED_*` entries only from `strong_quartiles + strong_pairs`, even though its priority list explicitly contains `LOCALIZED_Q2_Q3_Q4`. The development plan consequently lists only `COMPLEMENTARY_Q3_Q4`, `LOCALIZED_Q3_Q4`, and `FULL_DEPTH_ONLY`. The validation candidate calculation repeats the same omission, so `final_opening_v35.py` selected `FULL_DEPTH_ONLY`. `adjudicate_v35.py:patterns()` repeats it again, producing `no_shared_smaller_organization_passed=true`.

This is an **implementation/adjudication omission**, propagated into report wording, not an intentionally excluded mechanism class. A read-only replay of the existing gate, with the already-predeclared triple admitted, puts `LOCALIZED_Q2_Q3_Q4` in the development and validation intersection for both models. It would have preceded `FULL_DEPTH_ONLY` in the frozen `FINAL_PRIORITY`. The existing independent-final response, however, tested `FULL_DEPTH_ONLY` only; it cannot confirm the triple. No V35 final data are reopened here.

## Amendment and scientific boundary

The historical sentence “no shared strong subset” is false for the measured Q2+Q3+Q4 strong-gate subset. The defensible correction is: **a shared three-quartile subset passed the development and validation strong gates, but it was omitted from the finalist construction and never tested as the V35 independent-final mechanism.** Accordingly, V35 does not establish that no smaller read-side organization qualified for validation, nor does it establish independent-final confirmation of a hierarchy. V35-F/H cannot be retroactively promoted to confirmed. V35-I remains unconfirmed. Other negative V35 mechanism outcomes are not automatically overturned by this correction.

V36 will use neither the erroneous “no shared subset” claim nor an unearned final confirmation as a premise. Its local operator experiments require fresh panels and separate prospective gates.

## Source integrity

All checks used repository revision `9c272d501f6c0beb6ee0bf6c643dbdc77aef9475`. SHA-256 of the key sealed inputs:

| Input | SHA-256 |
|---|---|
| `configs/hierarchical_read_v35.yaml` | `65ca8e5254be7b1a2949a46f333c994869e7664d23db1609e27267725e7d4cf0` |
| `results/v35/processed/depth_analysis_development_v35.json` | `bbdf6400d3ac86873995ce9d0c7beb7e2a7b598299dcae5f1ae6304c433f6ae4` |
| `results/v35/processed/depth_analysis_validation_v35.json` | `0a6b7dd7b464d48c069a6da907b38eff28867784a0780a44a50db739243cda3d` |
| `results/v35/processed/development_plan_v35.json` | `62e5373715440e48fc7dc73edc775af47580212088053026c82293cd8e8ac7d8` |
| `results/v35/processed/final_opening_v35.json` | `48a88bf9a8e229f7dac1de1b03dd27cd46a31017b710e0401b898869028b8154` |
| `results/v35/processed/v35_adjudication.json` | `1fa4ba3f168dad0c348ba818314e0b8b73fcf5d7620b25f672d47a928ff7880a` |

No V1–V35 file was modified for this audit.
