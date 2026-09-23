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
