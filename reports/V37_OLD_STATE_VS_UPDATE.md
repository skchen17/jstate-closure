# V37 旧状态与当前更新

Qwen TRUE_UPDATE/`delta` 依赖旧 S，因此将其与旧状态从不同自然来源拼接并不构成干净的 OLD_ONLY/UPDATE_ONLY。Falcon 固定 h/Conv 的 dBx 独立于旧 S，但 V37 未完整执行双向 OLD/UPDATE 因子未来端点；不判定 OLD_STATE_DOMINANT 或 CURRENT_UPDATE_DOMINANT。
