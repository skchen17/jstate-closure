# V34 功能阶段映射

|功能阶段|Qwen 原生对象|Falcon 原生对象|正式身份|
|---|---|---|---|
|TRUE_UPDATE|true delta / rank-one write|true dBx|写侧，可独立写入|
|TRANSFORMED_CONTROL|transformed g / beta|transformed dt / decay|写侧，可独立写入|
|POSTCONV_INPUT|post-Conv q/k/v|post-Conv x/B/C|写侧，可独立写入|
|RECURRENT_READ|recurrent mixer output|SSM mixer output|读侧，可独立写入|
|RESIDUAL_INTEGRATION|same mixer contribution at residual add|same mixer contribution at residual add|与读出节点别名，无法独立识别|

比较的是功能角色，不是张量名称、形状或数学公式。Qwen 与 Falcon 使用不同的循环微代数；24 个循环层均进入全深度正式干预。第五阶段在两个实装中是第四阶段输出直接进入残差加法，没有中间可独立覆盖的对象。
