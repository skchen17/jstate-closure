# V34 All Reports

本文件逐一收录 V34 全部 21 份独立报告；各标题下内容与同名文件一致。


---

<!-- V34_REPLICATION_RECONFIRMATION.md -->

# V34 高层效应重确认

|角色|模型|n|正收益率|中位修正|中位对齐|家族|门槛|
|---|---|---|---|---|---|---|---|
|开发|Q|80|1.000|0.395|0.804|5|通过|
|开发|F|80|0.975|0.398|0.801|5|通过|
|验证|Q|40|1.000|0.386|0.798|5|通过|
|验证|F|40|1.000|0.348|0.762|5|通过|
|独立最终|Q|40|1.000|0.384|0.791|5|通过|
|独立最终|F|40|1.000|0.330|0.750|5|通过|

开发集先在两模型通过 ≥80% 正收益、中位修正 ≥0.20、中位对齐 ≥0.50、≥4/5 家族门槛，才开启正式内部中介。验证和独立最终集的高层效应随后分别重现。定义沿用 V32/V33，未重调。

---

<!-- V34_FUNCTIONAL_STAGE_MAPPING.md -->

# V34 功能阶段映射

|功能阶段|Qwen 原生对象|Falcon 原生对象|正式身份|
|---|---|---|---|
|TRUE_UPDATE|true delta / rank-one write|true dBx|写侧，可独立写入|
|TRANSFORMED_CONTROL|transformed g / beta|transformed dt / decay|写侧，可独立写入|
|POSTCONV_INPUT|post-Conv q/k/v|post-Conv x/B/C|写侧，可独立写入|
|RECURRENT_READ|recurrent mixer output|SSM mixer output|读侧，可独立写入|
|RESIDUAL_INTEGRATION|same mixer contribution at residual add|same mixer contribution at residual add|与读出节点别名，无法独立识别|

比较的是功能角色，不是张量名称、形状或数学公式。Qwen 与 Falcon 使用不同的循环微代数；24 个循环层均进入全深度正式干预。第五阶段在两个实装中是第四阶段输出直接进入残差加法，没有中间可独立覆盖的对象。

---

<!-- V34_EXACT_INSTRUMENTATION_AUDIT.md -->

# V34 精确仪器审计

|模型|校准层数|原生重放 bitwise|校准干预次数|
|---|---|---|---|
|Q|24|True|10|
|F|24|True|10|

校准使用未进入正式集的状态；每一正式阶段两方向都在每个探针与层记录原生、请求、实现哈希。正式 Parquet 位于 `results/v34/processed/mediation_audit_*_v34.parquet`；完整精度与拓扑审计以原始记录为准。

---

<!-- V34_UPDATE_TERM_MEDIATION.md -->

# V34 TRUE_UPDATE 中介

|阶段|角色|模型|REM|REST|余弦|家族|判定|
|---|---|---|---|---|---|---|---|
|TRUE_UPDATE|开发|Q|0.490|-0.864|-0.327|0|TESTED|
|TRUE_UPDATE|开发|F|0.077|0.036|0.230|0|TESTED|
|TRUE_UPDATE|验证|Q|0.451|-0.844|-0.343|0|TESTED|
|TRUE_UPDATE|验证|F|0.032|0.050|0.236|0|TESTED|

真实更新项只覆盖当前 token 的写入贡献，不等于完整新 REC 状态。Qwen 的反向插入在独立开发、验证中反而恶化供体误差；不能由重构 outgoing state 推出它是中介。

---

<!-- V34_GATE_DECAY_MEDIATION.md -->

# V34 TRANSFORMED_CONTROL 中介

|阶段|角色|模型|REM|REST|余弦|家族|判定|
|---|---|---|---|---|---|---|---|
|TRANSFORMED_CONTROL|开发|Q|0.023|0.007|0.144|0|TESTED|
|TRANSFORMED_CONTROL|开发|F|0.027|0.032|0.200|0|TESTED|
|TRANSFORMED_CONTROL|验证|Q|0.028|0.021|0.185|0|TESTED|
|TRANSFORMED_CONTROL|验证|F|0.040|0.027|0.193|0|TESTED|

使用核实际消费的变换后 Qwen g/beta 与 Falcon dt/decay，不是原始投影 logits；两模型双向效果均远低于强门槛。

冻结验证小子集的可分离子量：

|模型|原生子量|REM|REST|余弦|
|---|---|---|---|---|
|Q|POSTCONV_INPUT:k|0.016|-0.004|0.135|
|Q|POSTCONV_INPUT:q|0.239|0.124|0.474|
|Q|POSTCONV_INPUT:v|0.198|-0.296|-0.202|
|Q|TRANSFORMED_CONTROL:beta|0.020|0.013|0.154|
|Q|TRANSFORMED_CONTROL:g|0.000|-0.045|-0.053|
|F|POSTCONV_INPUT:B|-0.019|0.027|0.120|
|F|POSTCONV_INPUT:C|-0.008|0.112|0.204|
|F|POSTCONV_INPUT:x|0.213|0.179|0.425|

Falcon dt 与由其确定的 decay 是耦合变量，未把派生 decay 伪装成独立控制点。子量实验是探索性，不重定义正式联合阶段。

---

<!-- V34_POSTCONV_INPUT_MEDIATION.md -->

# V34 POSTCONV_INPUT 中介

|阶段|角色|模型|REM|REST|余弦|家族|判定|
|---|---|---|---|---|---|---|---|
|POSTCONV_INPUT|开发|Q|0.251|0.031|0.247|0|TESTED|
|POSTCONV_INPUT|开发|F|0.323|0.241|0.496|0|TESTED|
|POSTCONV_INPUT|验证|Q|0.218|0.030|0.304|0|TESTED|
|POSTCONV_INPUT|验证|F|0.317|0.213|0.502|0|TESTED|

使用短 Conv 之后进入循环算子的确切输入，而不是 Conv 前投影。它们可双向精确写入，但恢复比例与方向未达到强门槛。

冻结验证小子集的 Qwen q/k/v、Falcon x/B/C 单独替换：

|模型|原生子量|REM|REST|余弦|
|---|---|---|---|---|
|Q|POSTCONV_INPUT:k|0.016|-0.004|0.135|
|Q|POSTCONV_INPUT:q|0.239|0.124|0.474|
|Q|POSTCONV_INPUT:v|0.198|-0.296|-0.202|
|Q|TRANSFORMED_CONTROL:beta|0.020|0.013|0.154|
|Q|TRANSFORMED_CONTROL:g|0.000|-0.045|-0.053|
|F|POSTCONV_INPUT:B|-0.019|0.027|0.120|
|F|POSTCONV_INPUT:C|-0.008|0.112|0.204|
|F|POSTCONV_INPUT:x|0.213|0.179|0.425|

单独子量是探索性，不能用最优子量替换预定联合阶段门槛。

---

<!-- V34_RECURRENT_READ_MEDIATION.md -->

# V34 RECURRENT_READ 中介

|阶段|角色|模型|REM|REST|余弦|家族|判定|
|---|---|---|---|---|---|---|---|
|RECURRENT_READ|开发|Q|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|开发|F|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|验证|Q|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|验证|F|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|独立最终|Q|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|独立最终|F|1.000|1.000|1.000|5|通过|

全 24 层状态条件化 mixer 输出双向替换通过各角色、各模型、五类任务；这是读侧因果截面。全层输出复制可以形成宽截面，不能单凭 1.00 的效果定位内部写入算法。

---

<!-- V34_RESIDUAL_INTEGRATION_MEDIATION.md -->

# V34 RESIDUAL_INTEGRATION 中介

|阶段|角色|模型|REM|REST|余弦|家族|判定|
|---|---|---|---|---|---|---|---|
|RESIDUAL_INTEGRATION|开发|Q|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|
|RESIDUAL_INTEGRATION|开发|F|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|
|RESIDUAL_INTEGRATION|验证|Q|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|
|RESIDUAL_INTEGRATION|验证|F|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|

校准显示进入残差的最早可写循环贡献与 mixer 输出是同一个张量/节点；不存在更下游且仍只覆盖循环贡献的独立接口。故标记 ALIASED_NOT_SEPARATELY_IDENTIFIABLE，V34-F 不成立，不能重复计数 V34-E。

---

<!-- V34_MULTI_STAGE_PIPELINE_MEDIATION.md -->

# V34 多阶段流水线

开发和验证中，预定的 RECURRENT_READ 单阶段已在两模型通过。按冻结的条件规则，WRITE_PIPELINE、RECURRENT_PIPELINE、FULL_RECURRENCE_TO_RESIDUAL 不启动；没有测试结果，也没有声称分布式中介。

---

<!-- V34_MEDIATION_OVERLAP.md -->

# V34 中介重叠

只有一个可独立识别的功能阶段通过强门槛。写侧三个阶段均未通过，残差整合与读出为同一可写节点；因此没有两个合格独立阶段可做预定的 S1/S2/S1+S2 重叠分解。REM 值不相加，不给出 serial、redundant 或 complementary 分类。

---

<!-- V34_CONTEXT_SPECIFIC_MEDIATOR.md -->

# V34 读出阶段语境特异性

|模型|匹配恢复|错 token 恢复|REC-only 阶段恢复|Conv 条件差异率|供体 KV 恢复|
|---|---|---|---|---|---|
|Q|1.000|-5.735|-7.882|1.000|1.000|
|F|1.000|-11.706|-13.352|1.000|1.000|

每模型验证集五家族各两状态、同一 Conv 目标；错配值取同状态的另一冻结探针 token。错配控制有 off-manifold 风险，因此只作辅助证据，不独立证明语义匹配。

---

<!-- V34_CONV_DEPENDENCE.md -->

# V34 Conv 依赖

|模型|匹配恢复|错 token 恢复|REC-only 阶段恢复|Conv 条件差异率|供体 KV 恢复|
|---|---|---|---|---|---|
|Q|1.000|-5.735|-7.882|1.000|1.000|
|F|1.000|-11.706|-13.352|1.000|1.000|

比较 donor REC + donor Conv 与 donor REC + recipient Conv 的读出阶段值，以及它们在同一 Conv-only 目标上的反向插入。阶段哈希差异与恢复差异描述条件性；读出层是宽截面，不能反推出某个局部门控是唯一来源。

---

<!-- V34_REC_ONLY_MICRO_CONTROL.md -->

# V34 REC-only 微阶段对照

|模型|匹配恢复|错 token 恢复|REC-only 阶段恢复|Conv 条件差异率|供体 KV 恢复|
|---|---|---|---|---|---|
|Q|1.000|-5.735|-7.882|1.000|1.000|
|F|1.000|-11.706|-13.352|1.000|1.000|

REC-only 分支的阶段值并不等同于联合分支，且 REC-only 的高层供体误差见同角色 factorial Parquet。原生阶段值存在差异不等于足够的未来修正；只有精确替换后的响应方向与收益才进入中介结论。

---

<!-- V34_FUNCTIONAL_MEDIATION_PROFILE.md -->

# V34 功能中介画像

|功能阶段|角色|模型|REM|REST|余弦|家族|判定|
|---|---|---|---|---|---|---|---|
|TRUE_UPDATE|开发|Q|0.490|-0.864|-0.327|0|TESTED|
|TRUE_UPDATE|开发|F|0.077|0.036|0.230|0|TESTED|
|TRUE_UPDATE|验证|Q|0.451|-0.844|-0.343|0|TESTED|
|TRUE_UPDATE|验证|F|0.032|0.050|0.236|0|TESTED|
|TRANSFORMED_CONTROL|开发|Q|0.023|0.007|0.144|0|TESTED|
|TRANSFORMED_CONTROL|开发|F|0.027|0.032|0.200|0|TESTED|
|TRANSFORMED_CONTROL|验证|Q|0.028|0.021|0.185|0|TESTED|
|TRANSFORMED_CONTROL|验证|F|0.040|0.027|0.193|0|TESTED|
|POSTCONV_INPUT|开发|Q|0.251|0.031|0.247|0|TESTED|
|POSTCONV_INPUT|开发|F|0.323|0.241|0.496|0|TESTED|
|POSTCONV_INPUT|验证|Q|0.218|0.030|0.304|0|TESTED|
|POSTCONV_INPUT|验证|F|0.317|0.213|0.502|0|TESTED|
|RECURRENT_READ|开发|Q|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|开发|F|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|验证|Q|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|验证|F|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|独立最终|Q|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|独立最终|F|1.000|1.000|1.000|5|通过|
|RESIDUAL_INTEGRATION|开发|Q|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|
|RESIDUAL_INTEGRATION|开发|F|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|
|RESIDUAL_INTEGRATION|验证|Q|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|
|RESIDUAL_INTEGRATION|验证|F|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|

REM/REST 以 REC 在 Conv-only 之上的修正收益为分母；恢复余弦对比插入引出的修正向量与真实 Y11−Y01。只比较无量纲画像，不比较跨模型原始张量范数。

---

<!-- V34_CROSS_MODEL_MEDIATOR_COMPARISON.md -->

# V34 跨模型中介比较

|功能阶段|角色|模型|REM|REST|余弦|家族|判定|
|---|---|---|---|---|---|---|---|
|TRUE_UPDATE|开发|Q|0.490|-0.864|-0.327|0|TESTED|
|TRUE_UPDATE|开发|F|0.077|0.036|0.230|0|TESTED|
|TRUE_UPDATE|验证|Q|0.451|-0.844|-0.343|0|TESTED|
|TRUE_UPDATE|验证|F|0.032|0.050|0.236|0|TESTED|
|TRANSFORMED_CONTROL|开发|Q|0.023|0.007|0.144|0|TESTED|
|TRANSFORMED_CONTROL|开发|F|0.027|0.032|0.200|0|TESTED|
|TRANSFORMED_CONTROL|验证|Q|0.028|0.021|0.185|0|TESTED|
|TRANSFORMED_CONTROL|验证|F|0.040|0.027|0.193|0|TESTED|
|POSTCONV_INPUT|开发|Q|0.251|0.031|0.247|0|TESTED|
|POSTCONV_INPUT|开发|F|0.323|0.241|0.496|0|TESTED|
|POSTCONV_INPUT|验证|Q|0.218|0.030|0.304|0|TESTED|
|POSTCONV_INPUT|验证|F|0.317|0.213|0.502|0|TESTED|
|RECURRENT_READ|开发|Q|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|开发|F|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|验证|Q|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|验证|F|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|独立最终|Q|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|独立最终|F|1.000|1.000|1.000|5|通过|
|RESIDUAL_INTEGRATION|开发|Q|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|
|RESIDUAL_INTEGRATION|开发|F|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|
|RESIDUAL_INTEGRATION|验证|Q|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|
|RESIDUAL_INTEGRATION|验证|F|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|

两模型在独立开发、验证以及最终集均呈读侧通过、写侧未通过的 PROFILE MATCH。该结果支持同一宽功能截面，不证明 Qwen delta 与 Falcon dBx 是同一算法。这是对全 24 个循环层读出贡献的宽因果截面，不证明两模型采用相同的更新代数，也不能把与其别名的残差整合点另算一个独立中介。

---

<!-- V34_DEPTH_LOCALIZATION.md -->

# V34 相对深度定位

|模型|相对深度组|正收益状态|REM 中位|REST 中位|余弦中位|
|---|---|---|---|---|---|
|Q|full_depth|5|1.000|1.000|1.000|
|Q|quarter_1|5|0.433|0.283|0.495|
|Q|quarter_2|5|0.288|0.146|0.425|
|Q|quarter_3|5|0.481|0.151|0.593|
|Q|quarter_4|5|0.572|0.489|0.593|
|F|full_depth|5|1.000|1.000|1.000|
|F|quarter_1|5|0.116|0.196|0.589|
|F|quarter_2|5|0.141|0.120|0.301|
|F|quarter_3|5|0.586|0.520|0.675|
|F|quarter_4|5|0.531|0.419|0.589|

通过读出阶段后才进行该探索性分析，按每模型 24 个循环层的相对四分位划分，验证集每家族一状态；不把原始层号当跨架构同源层，也不据 5 状态细分精确层位。

---

<!-- V34_MULTI_HORIZON_MEDIATION.md -->

# V34 多步传播

|模型|h|正收益状态|REM 中位|REST 中位|余弦中位|
|---|---|---|---|---|---|
|Q|2|5|0.652|0.701|0.779|
|Q|4|5|0.311|0.164|0.346|
|F|2|4|0.513|0.203|0.475|
|F|4|4|0.193|0.069|0.254|

仅 h1 执行读出补丁，后续 token 的 h2/h4 自然演化；冻结验证子集每家族一状态、每状态用前四个预选探针作为 token 序列。这是探索性小样本传播分析，不替代 h1 正式门槛。

---

<!-- V34_STRICT_INTERFACE_AUDIT.md -->

# V34 严格接口审计

|功能阶段|Qwen 原生对象|Falcon 原生对象|正式身份|
|---|---|---|---|
|TRUE_UPDATE|true delta / rank-one write|true dBx|写侧，可独立写入|
|TRANSFORMED_CONTROL|transformed g / beta|transformed dt / decay|写侧，可独立写入|
|POSTCONV_INPUT|post-Conv q/k/v|post-Conv x/B/C|写侧，可独立写入|
|RECURRENT_READ|recurrent mixer output|SSM mixer output|读侧，可独立写入|
|RESIDUAL_INTEGRATION|same mixer contribution at residual add|same mixer contribution at residual add|与读出节点别名，无法独立识别|

校准原生重放、双向精确写回、未动 recipient KV、同背景/同探针/同位置都由机器记录核查；错误会使整次运行失败。阶段 5 的别名状态是接口限制，不以整个残差向量替代循环贡献。原始哈希和每探针 24 层证据见 interface_audit 与 mediation_audit 文件。

---

<!-- V34_EXECUTION_MANIFEST.md -->

# V34 执行清单

|模型|revision|权重 SHA-256|
|---|---|---|
|Q|851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a|26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61, cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188|
|F|06d7330266253c20784f58b0a846dc09a9d12cee|9acd2ef3e946e88a6e8cb14d38942f2ffbf6f52a5bc37312c1e0cc64574f1aeb|

父提交 `b8ee1284e72ec4a79d64b624a189de2ca57d2278`；基础协议 freeze `afeb71a2fef26ad1ac5a4066e127f1e90861079fa813906b696b50c02ec507aa`；最终开放 freeze `85bfc32dbad14f2295fc401d2f1ce8557d56567070fb93716e43ea453af06422`；裁决 freeze `002684c5d17759bd4ebd20732d441353e44b30db276fd3f5a282f76855d3d1bc`。面板：每模型 20 校准、80 开发、40 验证、40 独立最终，跨模型语义配对并排除 V28–V33 正式状态。精确 tensor 原值未入 Git；提交机器 JSON、Parquet、NPZ、哈希及报告。完整文件清单见 `results/v34/processed/v34_integrity_index.json`。

---

<!-- V34_SCIENTIFIC_ANSWERS.md -->

# V34 科学问题逐项回答

1. 高层修正再次复现：是，两模型三角色均通过。

2–5. post-Conv、变换控制、真实更新、循环读出：均可双向精确写入。

6. 残差整合的循环贡献与读出同节点；无法独立识别。

7–10. 各阶段 REM、REST、余弦和 ≥50% 判定见功能画像；只有读出通过。

11–16. 同一读侧截面通过两模型；真实更新、门控、post-Conv 输入及独立残差阶段未通过。

17. 单阶段已通过，预定多阶段流水线无需启动；不声称分布式中介。

18. 两模型无量纲画像同为 read-side dominant。

19–21. 匹配/错配、Conv 条件性、REC-only 与 KV 结果见冻结小子集报告；错配有 off-manifold 限制。

22–23. 相对深度与 h2/h4 是通过阶段后的探索性小子集分析，详见相应报告。

24–28. V34-A/E 成立；B/C/D/F/G/H/I 不成立或未启动，见裁决 JSON。

29. 应用实验获授权，但 V34 未建立实际应用收益。

30. 第三模型复制获授权，但当前结论只覆盖这两个模型。

这是对全 24 个循环层读出贡献的宽因果截面，不证明两模型采用相同的更新代数，也不能把与其别名的残差整合点另算一个独立中介。

---

<!-- V34_COMPLETE_REPORT.md -->

# V34 Complete Report

**Cross-Model Functional Mediation of REC–Conv Correction**

**V34-A 与 V34-E 成立**：两模型同一读出功能阶段在开发、验证和独立最终集上通过双向因果中介门槛。 这是对全 24 个循环层读出贡献的宽因果截面，不证明两模型采用相同的更新代数，也不能把与其别名的残差整合点另算一个独立中介。

## 高层重确认

|角色|模型|n|正收益率|中位修正|中位对齐|家族|门槛|
|---|---|---|---|---|---|---|---|
|开发|Q|80|1.000|0.395|0.804|5|通过|
|开发|F|80|0.975|0.398|0.801|5|通过|
|验证|Q|40|1.000|0.386|0.798|5|通过|
|验证|F|40|1.000|0.348|0.762|5|通过|
|独立最终|Q|40|1.000|0.384|0.791|5|通过|
|独立最终|F|40|1.000|0.330|0.750|5|通过|

## 功能阶段双向中介

|功能阶段|角色|模型|REM|REST|余弦|家族|判定|
|---|---|---|---|---|---|---|---|
|TRUE_UPDATE|开发|Q|0.490|-0.864|-0.327|0|TESTED|
|TRUE_UPDATE|开发|F|0.077|0.036|0.230|0|TESTED|
|TRUE_UPDATE|验证|Q|0.451|-0.844|-0.343|0|TESTED|
|TRUE_UPDATE|验证|F|0.032|0.050|0.236|0|TESTED|
|TRANSFORMED_CONTROL|开发|Q|0.023|0.007|0.144|0|TESTED|
|TRANSFORMED_CONTROL|开发|F|0.027|0.032|0.200|0|TESTED|
|TRANSFORMED_CONTROL|验证|Q|0.028|0.021|0.185|0|TESTED|
|TRANSFORMED_CONTROL|验证|F|0.040|0.027|0.193|0|TESTED|
|POSTCONV_INPUT|开发|Q|0.251|0.031|0.247|0|TESTED|
|POSTCONV_INPUT|开发|F|0.323|0.241|0.496|0|TESTED|
|POSTCONV_INPUT|验证|Q|0.218|0.030|0.304|0|TESTED|
|POSTCONV_INPUT|验证|F|0.317|0.213|0.502|0|TESTED|
|RECURRENT_READ|开发|Q|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|开发|F|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|验证|Q|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|验证|F|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|独立最终|Q|1.000|1.000|1.000|5|通过|
|RECURRENT_READ|独立最终|F|1.000|1.000|1.000|5|通过|
|RESIDUAL_INTEGRATION|开发|Q|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|
|RESIDUAL_INTEGRATION|开发|F|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|
|RESIDUAL_INTEGRATION|验证|Q|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|
|RESIDUAL_INTEGRATION|验证|F|—|—|—|—|ALIASED_NOT_SEPARATELY_IDENTIFIABLE|

判据相对于 REC 对 Conv-only 的附加修正收益，而非完整供体效应；所有正式写入保持同一 token/位置、原生 recipient KV、六个冻结探针，逐次记录 requested/realized 哈希。 Stage 5 is aliased with Stage 4 and is not counted separately. The frozen multi-stage sets were not run because a single stage qualified.

## 冻结小子集与边界

|模型|匹配恢复|错 token 恢复|REC-only 阶段恢复|Conv 条件差异率|供体 KV 恢复|
|---|---|---|---|---|---|
|Q|1.000|-5.735|-7.882|1.000|1.000|
|F|1.000|-11.706|-13.352|1.000|1.000|

相对深度和 h2/h4 详见各自报告；它们均为探索性，不改变正式 h1 决策。

最终裁决：`V34-A_SHARED_FUNCTIONAL_MEDIATOR_IDENTIFIED, V34-E_SHARED_RECURRENT_READ_MEDIATION`。应用实验授权=True，第三模型复制授权=True；实际应用收益未建立。 机器证据在 `results/v34/processed/`；V1–V33 历史记录未修改。
