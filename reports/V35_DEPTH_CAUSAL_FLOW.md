# V35 深度边界因果流解释性轨迹

|模型|边界|张量|joint−Conv L2 中位|
|---|---|---|---|
|Q|Q1|read|0.874|
|Q|Q1|residual|1.075|
|Q|Q2|read|0.726|
|Q|Q2|residual|1.693|
|Q|Q3|read|1.000|
|Q|Q3|residual|2.932|
|Q|Q4|read|2.650|
|Q|Q4|residual|6.617|
|F|Q1|read|0.070|
|F|Q1|residual|2.737|
|F|Q2|read|0.229|
|F|Q2|residual|6.807|
|F|Q3|read|1.281|
|F|Q3|residual|16.991|
|F|Q4|read|0.312|
|F|Q4|residual|31.854|

验证集每家族一个冻结状态、首个冻结探针，记录 recipient、Conv-only、joint 在四个边界的残差与原生读出哈希及差异。轨迹仅解释已通过的替换实验；原始 L2 不跨模型比较，也不作为正式中介证据。
