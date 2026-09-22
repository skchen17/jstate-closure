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
