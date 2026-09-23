# V37 All Reports — Computational Origin of REC–Conv Conditional Effects

本文件独立收录 V37 全部专题报告；V36 不在此计分。

## 文件 SHA-256

|报告|SHA-256|
|---|---|
|V37_COMPLETE_REPORT.md|611f1d0fccc79c5b762c4ed70322b2f2f490fec79102f9016aa7129ea6c63dbd|
|V37_CROSS_LAYER_OPERATOR_CONDITIONING.md|de7fdbf9f595b6dfa3ea07c23f671b1c73d1341b779bb20835948f9f38b5f0d1|
|V37_CROSS_MODEL_OPERATOR_LAW.md|0aabd57c073a596bc8662d76725041448d16eca4a4d0fb14c51ddb932ccb62b8|
|V37_DERIVED_VARIABLE_DEPENDENCY_AUDIT.md|ae0a669313447b1fd70e8261340be914276d30b5747e453ce92f1bfd38eb03e6|
|V37_EXECUTION_MANIFEST.md|e53b1998b9bdcef6ea3311e8bea5b6d4c1376635fccc60abc3a9fef6b1169bc0|
|V37_FROZEN_STARTING_POINT.md|a1bf1a03c6d88173c76c91ae56e7f6ec2d00355b302209cf5aca6139ea689ec9|
|V37_GAIN_ROTATION_CONTROLS.md|9e4ae13e19ab87158da4d46a5f486e03c63c5e4da40dd671a12e3f43db0926f0|
|V37_HIGH_LEVEL_RECONFIRMATION.md|e2bc2f26936e69e4df03c59f54713335d2d3329b0afa33286b925cebb7562459|
|V37_LOCAL_FIXED_INPUT_FACTORIAL.md|26d47b0f8250d4a9e4992fc5b6b75c91b26406c21cb03c2d2ebcb0ed7dc1bbab|
|V37_LOCAL_TO_GLOBAL_REINSERTION.md|0590708fe5ca0de81ca6145657e86a29feed3fc631ba7632ec13f2528607b13a|
|V37_NATIVE_RECURRENCE_EQUATIONS.md|dc773557f7cb207076840158f29937b195545cc4da36dc58f5e00f1d5f02b026|
|V37_NATIVE_TERM_DECOMPOSITION.md|b6f64b00103b5f85aaaef1f568f14899584048e027b7461b3dc5e96644947563|
|V37_OLD_STATE_VS_UPDATE.md|cbf5b08b17f6597f305cf506d7b7f387ad21353eb32b13181f749e3967c889bd|
|V37_OPERATOR_SENSITIVITY_GEOMETRY.md|e2af36ce6abadf56ca3455757e65fe6de6f550c2be90b29d6aed8d09d3c6f192|
|V37_PRE_POST_READ_TRANSFORMATION.md|8f2174754ff1623141c3ca8bb10c63aac15c32541f91419d2a80f81530f0ced5|
|V37_Q2_CONTEXT_CONDITIONING.md|fce8bc222d5d3361088f099471def4daac7a142865a0fa64adb7d4b23f50dc9a|
|V37_Q2_DIRECT_INDIRECT_DECOMPOSITION.md|79ad5905936826fb32288cf31d24d970ff49438c4b603c3eaef7bf50d6c102f4|
|V37_Q2_Q3_Q4_FRESH_REPLICATION.md|fb76693c5511f648e3f964ec18760ea40be212e49fdc4b9310e17d09e4cc34ae|
|V37_Q2_Q3_Q4_MECHANISTIC_RECONSTRUCTION.md|d2af772d4a429f64bc926973b00172bdc8dd8233d97f4fcd4640f26d1994317d|
|V37_READ_OPERATOR_DERIVATION.md|9433997f95dde1dc5223347febf3d5366e015f927f22e4643e55f65d02430032|
|V37_READ_OPERATOR_PREDICTION.md|35703eb57ea73707cb466f029ad52d6814057e318bfa2423df3605eeadc1ff2c|
|V37_SCIENTIFIC_ANSWERS.md|31586a9b16322f3ad7cdc86ee33a072f67a66299264eca92b347008490ce2c69|
|V37_STATE_OPERATOR_COMPATIBILITY_MATRIX.md|80b78028df6df6f251b5723423743a0bf5cf8694e6437b7c38c08d2181c03290|
|V37_STATE_VS_OPERATOR_COUNTERFACTUAL.md|f72001c5935147e58496643b2b7177e0fb467aec8506b3447a2b83196f55a6d2|
|V37_STRICT_INTERFACE_AUDIT.md|6ce6367406f9f1c8444221e1e92f17011a7d8168e470c4c7cb4ee2de2668fc6a|
|V37_TASK_GROUNDED_FUNCTIONAL_TEST.md|c755365441569c770cfb36257195b6288eb5394fb708d0ff974252cea201ece6|
|V37_THREE_BRANCH_DESIGN.md|65ead0e836c0ef5eecd0fb7d25e47734931ed315105916d55ee993a240f60004|
|V37_TRUE_UPDATE_REINTERPRETATION.md|b83258614569c1e4492a8c4fb052a1cf0ceb2f54272b7eab485562d98bbbfd52|
|V37_UPSTREAM_CONTEXT_DEPENDENCE.md|afbae23135a7b60a58d84ceba202d535290d4d4da8d0e897a62c96851fb2088b|

---

## V37_COMPLETE_REPORT.md

# V37 Complete Report — Computational Origin of REC–Conv Conditional Effects

结论：V37 以新独立样本重新确认高层 REC–Conv 条件效应和读出接口。最高支持层级为 `LEVEL_2_LOCAL_COMPUTATION`；局部读算子类通过开发和独立正式验证；跨层 Q2→后层下游排序的开发判据失败；局部原生计算等价不得提升为跨层或任务语义机制。已开放 `READ_OPERATOR_MATCHING`；独立最终两模型局部读算子门槛通过，详见 `final_analysis_v37.json`。

## 样本与先验封存

校准/开发/正式验证/独立最终为 20/80/40/40 states/model、五家族 4/16/8/8。正式验证池和独立最终池彼此独立、在正式 intervention outcome 前封存。V18 剩余 training-source 与 V36 正式结果均不计入 V37。

## 高层与接口

|角色|模型|状态|正效应|误差减少|家族|通过|
|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|80|0.988|0.309|5|True|
|development|Falcon-H1-1.5B|80|1.000|0.380|5|True|
|validation|Qwen3.5-4B|40|0.950|0.423|4|True|
|validation|Falcon-H1-1.5B|40|1.000|0.422|5|True|

|角色|模型|全读移除|全读恢复|Q234移除|Q234恢复|Q234家族|
|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|1.000|1.000|0.976|0.773|5|
|development|Falcon-H1-1.5B|1.000|1.000|0.924|0.929|5|
|validation|Qwen3.5-4B|1.000|1.000|0.998|0.771|5|
|validation|Falcon-H1-1.5B|1.000|1.000|0.884|0.924|5|

## 局部读算子与下游

|角色|模型|状态|局部余弦|幅度秩相关|成对排序|单层下游收益比例|局部门槛|
|---|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|80|1.000|1.000|1.000|0.027|True|
|development|Falcon-H1-1.5B|80|1.000|1.000|1.000|0.032|True|
|validation|Qwen3.5-4B|40|1.000|1.000|1.000|0.023|True|
|validation|Falcon-H1-1.5B|40|1.000|1.000|1.000|0.033|True|

|模型|状态|raw余弦|raw排序|mixer余弦|mixer排序|门槛|
|---|---|---|---|---|---|---|
|Qwen3.5-4B|40|1.000|1.000|1.000|1.000|True|
|Falcon-H1-1.5B|40|1.000|1.000|1.000|1.000|True|

|模型|局部余弦|下游排序|通过家族|跨层门槛|
|---|---|---|---|---|
|Qwen3.5-4B|1.000|0.500|1|False|
|Falcon-H1-1.5B|1.000|0.500|0|False|

|模型|Q234自然恢复比例|方向余弦|Q2条件增强|
|---|---|---|---|
|Qwen3.5-4B|0.668|0.861|0.421|
|Falcon-H1-1.5B|0.796|0.874|0.709|

## 对照和解释上界

|模型|阶段|解释|方向余弦|相对向量误差|相对幅度误差|
|---|---|---|---|---|---|
|Qwen3.5-4B|raw|native|1.000|0.000|0.000|
|Qwen3.5-4B|raw|gain|0.961|0.294|0.061|
|Qwen3.5-4B|raw|fixed_rotation|0.973|0.250|0.059|
|Qwen3.5-4B|mixer|native|1.000|0.000|0.000|
|Qwen3.5-4B|mixer|gain|0.959|0.302|0.072|
|Qwen3.5-4B|mixer|fixed_rotation|0.962|0.293|0.071|
|Falcon-H1-1.5B|raw|native|1.000|0.000|0.000|
|Falcon-H1-1.5B|raw|gain|0.994|0.171|0.088|
|Falcon-H1-1.5B|raw|fixed_rotation|0.994|0.164|0.089|
|Falcon-H1-1.5B|mixer|native|1.000|0.000|0.000|
|Falcon-H1-1.5B|mixer|gain|0.985|0.255|0.123|
|Falcon-H1-1.5B|mixer|fixed_rotation|0.984|0.254|0.122|

两模型新校准各 3240 等价行，最大误差零；但 source-derived 公式对 source-derived 运行精确属于 LEVEL 2 计算核验。单层自然下游恢复、Q2 条件化和语义任务问题分别单独判定，不由局部余弦代替。旧状态/更新拆分和 post-read 双向中介未完整执行，结论保持未判定。

---

## V37_CROSS_LAYER_OPERATOR_CONDITIONING.md

# V37 跨层算子条件化

|模型|局部余弦|下游排序|通过家族|跨层门槛|
|---|---|---|---|---|
|Qwen3.5-4B|1.000|0.500|1|False|
|Falcon-H1-1.5B|1.000|0.500|0|False|

局部 mixer 变化余弦约 1 是原生公式局部等价；真正的前瞻六 probe 下游排序没有过门槛。验证/最终不为失败的跨层候选打开，结论为开发集未证实，而非跨层作用不存在。

---

## V37_CROSS_MODEL_OPERATOR_LAW.md

# V37 跨模型算子法则

|角色|模型|状态|局部余弦|幅度秩相关|成对排序|单层下游收益比例|局部门槛|
|---|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|80|1.000|1.000|1.000|0.027|True|
|development|Falcon-H1-1.5B|80|1.000|1.000|1.000|0.032|True|
|validation|Qwen3.5-4B|40|1.000|1.000|1.000|0.023|True|
|validation|Falcon-H1-1.5B|40|1.000|1.000|1.000|0.033|True|

Qwen 与 Falcon 分别用自身原生方程实现同一个抽象：固定输入下旧状态经 Conv 条件化读映射；微观公式不同。共同局部门槛与跨层门槛分开报告，不能把两模型局部等价改写为共享全局修正算法。

---

## V37_DERIVED_VARIABLE_DEPENDENCY_AUDIT.md

# V37 派生变量依赖审计

Qwen：`(h,Conv)→q,k,v`，`h→G,β,z`，`(S,G,k)→m=kᵀGS`，`(v,m,β)→delta`，`(S,delta,k)→S'→r→norm→mixer`。`delta/TRUE_UPDATE` 依赖旧 S，不是独立于旧状态的新写入。

Falcon：`(h,Conv)→x,B,C`，`h→dt,dA,gate`，`(dt,B,x)→dBx`，`(S,dA,dBx)→S'→r→norm→mixer`。固定 h/Conv 时 dBx 独立于旧 S。缓存写入和输出均由原生重算，不把历史派生张量独立拼接。

---

## V37_EXECUTION_MANIFEST.md

# V37 执行与完整性清单

父提交 `d7b93f041706a60895f0294c9e75c9cb6363a4f6`；V37 基础封存 `8f00ae8001fe5872c70bb64fbbd35d158df5864c168fcbf7d8bb6f517b03383f`。四池 20/80/40/40 states/model，validation 和 independent final 都是预封存新池；V18 train-source 不进入 formal roles。模型权重/配置/tokenizer hash 与安装源码审计在 `artifacts/computational_origin_v37.freeze.json`、`source_audit_v37.json`，每一步的输入 SHA 和结果 SHA 在独立 stage freeze。独立最终开放 `True`，候选 `READ_OPERATOR_MATCHING`；V36 历史未改动。

---

## V37_FROZEN_STARTING_POINT.md

# V37 冻结起点

V37 父提交 `d7b93f041706a60895f0294c9e75c9cb6363a4f6` 为已推送 V36；V1–V36 正式文件和结果完全不变，V36 正式观测不计入 V37。仅复用经新校准审计的工程 instrumentation/方程。

新样本池：校准 `20`、开发 `80`、正式验证 `40`、独立最终 `40` states/model。验证和最终各有独立预封存池；V18 training-source 状态不进入 V37 formal validation/final。五任务族各为 4/16/8/8 states。池封存时两模型响应均未观测。样本为同任务模板的新程序/提示，不宣称任务族分布外泛化。

样本分组哈希：`{"calibration": "607055d330875b4a2be11f5e18e3a05163214cc25587797ff266d79e9e7ad117", "development": "624ed4fc15b5db01560489de777eba891080c6e178b500efcfb78b86f0262381", "independent_final": "f3d4a7839755f2321dbab49ab00726713062a2f9f4d4eaca3092d634122615b3", "validation": "687a97b08ef1619cfc51a5cafdccf10b454b9ef8ee3d3341378944a89381029a"}`。源审计：`fa8dfc684d9ff5772f6c7b6248884ae1d016e0da378c6694e50c6124a0433be8`。

---

## V37_GAIN_ROTATION_CONTROLS.md

# V37 标量增益与冻结旋转对照

|模型|阶段|解释|方向余弦|相对向量误差|相对幅度误差|
|---|---|---|---|---|---|
|Qwen3.5-4B|raw|native|1.000|0.000|0.000|
|Qwen3.5-4B|raw|gain|0.961|0.294|0.061|
|Qwen3.5-4B|raw|fixed_rotation|0.973|0.250|0.059|
|Qwen3.5-4B|mixer|native|1.000|0.000|0.000|
|Qwen3.5-4B|mixer|gain|0.959|0.302|0.072|
|Qwen3.5-4B|mixer|fixed_rotation|0.962|0.293|0.071|
|Falcon-H1-1.5B|raw|native|1.000|0.000|0.000|
|Falcon-H1-1.5B|raw|gain|0.994|0.171|0.088|
|Falcon-H1-1.5B|raw|fixed_rotation|0.994|0.164|0.089|
|Falcon-H1-1.5B|mixer|native|1.000|0.000|0.000|
|Falcon-H1-1.5B|mixer|gain|0.985|0.255|0.123|
|Falcon-H1-1.5B|mixer|fixed_rotation|0.984|0.254|0.122|

标量增益与 rank-16 固定正交旋转均只在 development 的局部状态差上拟合，按模型×层位×阶段×目标算子冻结；validation 不重拟合。比较原生预测、增益和旋转的方向/向量误差/幅度误差。若简单对照接近原生，不能宣称复杂算子几何是独特解释。

---

## V37_HIGH_LEVEL_RECONFIRMATION.md

# V37 高层条件效应重新验证

|角色|模型|状态|正效应|误差减少|家族|通过|
|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|80|0.988|0.309|5|True|
|development|Falcon-H1-1.5B|80|1.000|0.380|5|True|
|validation|Qwen3.5-4B|40|0.950|0.423|4|True|
|validation|Falcon-H1-1.5B|40|1.000|0.422|5|True|

六 probe 先在同一状态内合并，state 是 bootstrap 独立单位；验证池绝非 V18 剩余 training-source。

---

## V37_LOCAL_FIXED_INPUT_FACTORIAL.md

# V37 固定输入 3×3 因子实验

|角色|模型|状态|局部余弦|幅度秩相关|成对排序|单层下游收益比例|局部门槛|
|---|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|80|1.000|1.000|1.000|0.027|True|
|development|Falcon-H1-1.5B|80|1.000|1.000|1.000|0.032|True|
|validation|Qwen3.5-4B|40|1.000|1.000|1.000|0.023|True|
|validation|Falcon-H1-1.5B|40|1.000|1.000|1.000|0.033|True|

A/B/C 为同前缀的三个自然 token，固定 Conv-only 轨迹的 h，逐格原生重算 `S_i×Conv_j`，不复制 donor mixer。逐 probe 预测先封存，后执行九格并核对请求/实现哈希。表中单层下游收益比例与局部代数门槛分列，不能互换。

---

## V37_LOCAL_TO_GLOBAL_REINSERTION.md

# V37 局部输出自然下游重插入

|角色|模型|状态|局部余弦|幅度秩相关|成对排序|单层下游收益比例|局部门槛|
|---|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|80|1.000|1.000|1.000|0.027|True|
|development|Falcon-H1-1.5B|80|1.000|1.000|1.000|0.032|True|
|validation|Qwen3.5-4B|40|1.000|1.000|1.000|0.023|True|
|validation|Falcon-H1-1.5B|40|1.000|1.000|1.000|0.033|True|

四格 AA/BA/AB/BB 的本地重算 mixer 在指定单层插入 Conv-only 支路，后层自然传播；其恢复比例只是单层效应，未将其加总成全深度机制。分母绝对收益低于 1.0 时比例缺失，只报告绝对量。

---

## V37_NATIVE_RECURRENCE_EQUATIONS.md

# V37 原生递推方程与等价校准

Qwen 单 token 实数形式：`S_d=G S; delta=β(v-kᵀ S_d); S'=S_d+k⊗delta; r=qᵀS'`。Falcon：`S'=dA⊙S+dBx; r=CᵀS'+Dx`。实际比较采用安装源码的 dtype/门控/归一化/输出投影顺序，不把代数式当成 BF16 逐位等式。

V37 新校准：Q/F 每模型 `20`/`20` 个状态，各 `3240`/`3240` 个 state×site×cell×stage 等价审计行；最大绝对误差 `0.0`/`0.0`，instrumented logits/cache/endpoints 均 bitwise。V36 旧校准没有替代此步。

---

## V37_NATIVE_TERM_DECOMPOSITION.md

# V37 原生项分解

Qwen 固定输入下 `r=qᵀG S - qᵀ[k⊗β(kᵀGS)] + qᵀ[k⊗βv]`：旧状态保持项、旧状态依赖的移除项、当前输入写入项。Falcon 固定输入下 `r=Cᵀ(dA⊙S)+CᵀdBx+Dx`。

|模型|校准行|旧状态读出差范数|更新读出差范数|更新独立比例|
|---|---|---|---|---|
|Qwen3.5-4B|2160|0.005|0.002|0.000|
|Falcon-H1-1.5B|2160|397.193|0.000|1.000|

这些是 V37 新校准池的依赖一致代数项，不是三项被独立操控后的行为因果结论。V37 未对每项逐项做完整双向整模型干预，因此旧/更新的正式优势类别不确认。

---

## V37_OLD_STATE_VS_UPDATE.md

# V37 旧状态与当前更新

Qwen TRUE_UPDATE/`delta` 依赖旧 S，因此将其与旧状态从不同自然来源拼接并不构成干净的 OLD_ONLY/UPDATE_ONLY。Falcon 固定 h/Conv 的 dBx 独立于旧 S，但 V37 未完整执行双向 OLD/UPDATE 因子未来端点；不判定 OLD_STATE_DOMINANT 或 CURRENT_UPDATE_DOMINANT。

---

## V37_OPERATOR_SENSITIVITY_GEOMETRY.md

# V37 算子敏感性几何

|模型|校准算子行|相对A算子余弦|相对A范数|
|---|---|---|---|
|Qwen3.5-4B|2160|0.987|1.000|
|Falcon-H1-1.5B|2160|0.988|1.002|

奇异谱、token 分支、site 和 family 标注在 calibration-only Parquet/NPZ。这里是 exact raw-state 线性映射的解析几何；没有前瞻因果低秩压缩检验，故不称低秩功能机制。

---

## V37_PRE_POST_READ_TRANSFORMATION.md

# V37 读前与读后转换

V37 在固定输入九格记录 raw、normalized、mixer 的交互范数、方向和自然轨迹对比，并重插入 mixer 检查下游。单凭读后交互放大或旋转不足以判定该阶段中介；没有独立、双向的 norm/gate-only 整模型干预，故 POST_READ_TRANSFORMATION 不确认。

---

## V37_Q2_CONTEXT_CONDITIONING.md

# V37 Q2 条件化实验

|模型|局部余弦|下游排序|通过家族|跨层门槛|
|---|---|---|---|---|
|Qwen3.5-4B|1.000|0.500|1|False|
|Falcon-H1-1.5B|1.000|0.500|0|False|

只替换上游 Q2 读输出，后层状态读/网络自然执行；在后层目标 REC 状态干预前冻结预测。开发集下游排序未达到 ≥0.75、≥4/5 家族，故该候选未进入正式验证。

---

## V37_Q2_DIRECT_INDIRECT_DECOMPOSITION.md

# V37 Q2 直接/间接路径

已观察 Q2 介入对后层隐藏输入与读因子的影响，但未构造可分离的 Q2 直接和 Q3/Q4 间接路径双向因子干预。现有对比无法识别路径特异中介量，不报告 DIRECT_DOMINANT 或 INDIRECT_DOMINANT。

---

## V37_Q2_Q3_Q4_FRESH_REPLICATION.md

# V37 Q2/Q3/Q4 新样本复核

|角色|模型|全读移除|全读恢复|Q234移除|Q234恢复|Q234家族|
|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|1.000|1.000|0.976|0.773|5|
|development|Falcon-H1-1.5B|1.000|1.000|0.924|0.929|5|
|validation|Qwen3.5-4B|1.000|1.000|0.998|0.771|5|
|validation|Falcon-H1-1.5B|1.000|1.000|0.884|0.924|5|

完整读出移除/恢复属于接口检验；Q2+Q3+Q4 是单独标记的次级复核，不重开 V35/V36 的旧最终。

---

## V37_Q2_Q3_Q4_MECHANISTIC_RECONSTRUCTION.md

# V37 Q2/Q3/Q4 自然重建

|模型|Q234自然恢复比例|方向余弦|Q2条件增强|
|---|---|---|---|
|Qwen3.5-4B|0.668|0.861|0.421|
|Falcon-H1-1.5B|0.796|0.874|0.709|

只替换指定段的 REC 状态，后层自然传播，不复制 Q3/Q4 或任意未来输出。恢复比例和方向说明组合仍有功能效应，却不单独定位是哪一条算子条件化路径。

---

## V37_READ_OPERATOR_DERIVATION.md

# V37 状态读算子推导

固定 h/Conv，Qwen 的实数线性状态读映射为 `qᵀ[G I - βk(kᵀG)]S`；Falcon 为 `Cᵀ(dA⊙S)`。两者均是局部状态差对 raw read 的映射。V37 用真实有限精度方程分别算 3×3 预测，再与原生运行比较；归一化/门控和 output projection 另作后读阶段，不直接等同 raw 线性算子。

---

## V37_READ_OPERATOR_PREDICTION.md

# V37 前瞻读算子预测

|角色|模型|状态|局部余弦|幅度秩相关|成对排序|单层下游收益比例|局部门槛|
|---|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|80|1.000|1.000|1.000|0.027|True|
|development|Falcon-H1-1.5B|80|1.000|1.000|1.000|0.032|True|
|validation|Qwen3.5-4B|40|1.000|1.000|1.000|0.023|True|
|validation|Falcon-H1-1.5B|40|1.000|1.000|1.000|0.033|True|

|模型|状态|raw余弦|raw排序|mixer余弦|mixer排序|门槛|
|---|---|---|---|---|---|---|
|Qwen3.5-4B|40|1.000|1.000|1.000|1.000|True|
|Falcon-H1-1.5B|40|1.000|1.000|1.000|1.000|True|

开发、验证和获准后的独立最终预测分别先冻结，之后才观察对应 3×3 局部 outcome。门槛为余弦≥0.90、幅度秩相关≥0.70、成对排序≥0.80、≥4/5 家族且两模型各自通过；这是局部计算层级，不自动证实跨层功能。

---

## V37_SCIENTIFIC_ANSWERS.md

# V37 科学问题答复

1. 新 V37 高层条件效应：开发和正式验证见高层表。2. Q2/Q3/Q4 次级读接口：见 `|validation|Falcon-H1-1.5B|1.000|1.000|0.884|0.924|5|`。3. 两架构原生公式及依赖：见本版方程/依赖报告。4. 局部状态×算子效应：|角色|模型|状态|局部余弦|幅度秩相关|成对排序|单层下游收益比例|局部门槛|
|---|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|80|1.000|1.000|1.000|0.027|True|
|development|Falcon-H1-1.5B|80|1.000|1.000|1.000|0.032|True|
|validation|Qwen3.5-4B|40|1.000|1.000|1.000|0.023|True|
|validation|Falcon-H1-1.5B|40|1.000|1.000|1.000|0.033|True|。

5. 简单增益/固定旋转能否解释：见增益旋转对照表，结论限定在已测局部向量。6. Q2 条件化是否建立跨层机制：否，开发下游排序未过门槛。7. 全段自然重建：|模型|Q234自然恢复比例|方向余弦|Q2条件增强|
|---|---|---|---|
|Qwen3.5-4B|0.668|0.861|0.421|
|Falcon-H1-1.5B|0.796|0.874|0.709|。8. OLD/UPDATE、读后中介及任务语义：缺少对应双向/外部变量实验，不作正式确认。9. 独立最终：已开放 `READ_OPERATOR_MATCHING`；独立最终两模型局部读算子门槛通过，详见 `final_analysis_v37.json`。。10. V36 结果完全排除，未借用旧正式样本或旧结论。

---

## V37_STATE_OPERATOR_COMPATIBILITY_MATRIX.md

# V37 状态×算子兼容矩阵

九格局部 raw/mixer 预测和观测向量保存于分模型、分角色 NPZ（大张量仅保留工作区，Git 只记录哈希）。逐格自然缓存和局部输入 hash 在 Parquet。

|角色|模型|B对角最强比例|C对角最强比例|
|---|---|---|---|
|development|Qwen3.5-4B|0.403|0.389|
|development|Falcon-H1-1.5B|0.292|0.333|
|validation|Qwen3.5-4B|0.375|0.417|
|validation|Falcon-H1-1.5B|0.431|0.306|

对角最强是次级、描述性模式，不是主公式预测门槛；全局未来只对预先指定四格重插入，未对九格都执行未来因果比较。

---

## V37_STATE_VS_OPERATOR_COUNTERFACTUAL.md

# V37 状态与算子反事实

固定同一 h 的 A/B/C REC × A/B/C Conv 九格是主要同前缀天然反事实。下游四格亦自然重插入；不能把错配格的非自然组合直接解释为模型平时会采取的行为。独立 final 若开放，只检验已选的局部读算子类。

---

## V37_STRICT_INTERFACE_AUDIT.md

# V37 严格接口与阴性对照

V37 自己的校准对两模型各 20 states×6 sites×9 cells×3 stages，最大误差零，instrumentation bitwise。记录自然 Conv-only、REC-only、joint、全读出移除/恢复、同前缀 A/B/C 缓存错配，以及 development-only 增益/旋转。随机打乱、跨家族错配、独立 post-read 双向移除未覆盖，报告中不得视作已通过。

---

## V37_TASK_GROUNDED_FUNCTIONAL_TEST.md

# V37 任务语义功能测试

未完成外部定义的五类语义变量恢复实验；向量 donor 误差/读出恢复不等于正确答案、绑定值、图节点或状态转移变量被恢复。V37 最多确认已测的局部计算和接口层级，不宣称 LEVEL 4 TASK FUNCTION。

---

## V37_THREE_BRANCH_DESIGN.md

# V37 三个自然 token 分支

A=recipient、B=donor、C=同前缀第三自然 token。A/B/C 均来自预冻结、响应盲的 token triplet；各自的 REC 与 Conv 缓存形成九格。C 是自然同前缀错配对照，不是随机合成 activation；跨家族和随机错配不用于主要语义判定。

---

## V37_TRUE_UPDATE_REINTERPRETATION.md

# V37 V34 TRUE_UPDATE 边界重释

V34 旧结果保持原样。Qwen 中移植派生 TRUE_UPDATE 会带入另一旧状态计算的 m，属依赖错配；Falcon 的 dBx 不依赖旧状态，但单独读出仍可能受 C/门控/后层影响。V37 未通过新双向干预在 A–E 解释间裁决，不能把 V34 失败概括成当前更新无作用。

---

## V37_UPSTREAM_CONTEXT_DEPENDENCE.md

# V37 上游上下文依赖

|模型|局部余弦|下游排序|通过家族|跨层门槛|
|---|---|---|---|---|
|Qwen3.5-4B|1.000|0.500|1|False|
|Falcon-H1-1.5B|1.000|0.500|0|False|

Q2 输出改变后 Q3/Q4 因子和 hidden input，局部映射方向吻合，但未来排序未过冻结门槛。这是上游影响的证据，不等于上游变化足以解释跨层行为。
