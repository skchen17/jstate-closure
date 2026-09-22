# V34 严格接口审计

|功能阶段|Qwen 原生对象|Falcon 原生对象|正式身份|
|---|---|---|---|
|TRUE_UPDATE|true delta / rank-one write|true dBx|写侧，可独立写入|
|TRANSFORMED_CONTROL|transformed g / beta|transformed dt / decay|写侧，可独立写入|
|POSTCONV_INPUT|post-Conv q/k/v|post-Conv x/B/C|写侧，可独立写入|
|RECURRENT_READ|recurrent mixer output|SSM mixer output|读侧，可独立写入|
|RESIDUAL_INTEGRATION|same mixer contribution at residual add|same mixer contribution at residual add|与读出节点别名，无法独立识别|

校准原生重放、双向精确写回、未动 recipient KV、同背景/同探针/同位置都由机器记录核查；错误会使整次运行失败。阶段 5 的别名状态是接口限制，不以整个残差向量替代循环贡献。原始哈希和每探针 24 层证据见 interface_audit 与 mediation_audit 文件。
