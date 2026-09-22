# V34 精确仪器审计

|模型|校准层数|原生重放 bitwise|校准干预次数|
|---|---|---|---|
|Q|24|True|10|
|F|24|True|10|

校准使用未进入正式集的状态；每一正式阶段两方向都在每个探针与层记录原生、请求、实现哈希。正式 Parquet 位于 `results/v34/processed/mediation_audit_*_v34.parquet`；完整精度与拓扑审计以原始记录为准。
