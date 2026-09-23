# V34 TRUE_UPDATE 失败的 V36 重新解释

Qwen 的 `W=k⊗beta*(v-kᵀG S)` 依赖旧 S。只替换 W 而不替换 `G S`，既漏掉旧状态项，也让 donor 派生的移除项与 recipient 旧状态不一致；所以 V34 TRUE_UPDATE 单独失败不能证明更新无关。Falcon 的 `dBx=dt*B*x` 固定输入时与 S 无关，TRUE_UPDATE 交换没有携带 REC 差；REC 的条件作用通过 `S*dA` 被 C 读取。两种架构原因不同。

本版在校准与开发/验证的公式重算支持上述依赖关系；没有对 V34 全深度 TRUE_UPDATE 历史数据作追认或覆写，也未声称重跑了一个全深度更新干预。
