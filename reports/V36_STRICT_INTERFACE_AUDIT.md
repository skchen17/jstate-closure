# V36 严格接口与对照审计

控制 A：校准完整一 token 原生重放 bitwise；每模型 20 状态、四层、两 REC 状态的公式 raw read 零误差。B/C/I：正式 factorial 的 recipient REC、donor REC × recipient/donor Conv 与自然 joint。D/E：固定 h 的第三自然 token Conv 与 REC 缓存九格，写回哈希在 proof 记录。F：Qwen q/k、Falcon C 的打乱算子组件，仅作为非天然算子诊断。G：合成反号 ΔS 的代数检查，不冒称自然 token。H：自然供体 mixer output 放入 Conv-only 支路作为单层上界，不冒称机制。所有运行均保持 recipient-native KV 与同一六 probe。F/G 不在自然状态流形，不能拿它们的输出当因果任务结果。

|角色|模型|供体混合器上界恢复率|旧状态投影|状态依赖更新投影|打乱算子余弦|反号对向余弦|
|---|---|---|---|---|---|---|
|development|Qwen3.5-4B|0.039|1.205|-0.204|0.988|0.999|
|development|Falcon-H1-1.5B|0.064|0.998|0.000|0.996|0.998|
|validation|Qwen3.5-4B|0.037|1.210|-0.211|0.987|0.999|
|validation|Falcon-H1-1.5B|0.076|0.998|0.000|0.996|0.998|

请求/实现 REC、Conv、mixer 哈希、local h 哈希、完整自然支路响应哈希和独立最终封存记录分别见 `local_proof_*`、`controls_proof_*`、factorial audit 与 `final_opening_v36.json`。未把 activation similarity 作为机制判据。
