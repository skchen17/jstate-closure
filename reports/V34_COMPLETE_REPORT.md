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
