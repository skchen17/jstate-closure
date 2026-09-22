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
