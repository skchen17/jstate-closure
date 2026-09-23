# V37 算子敏感性几何

|模型|校准算子行|相对A算子余弦|相对A范数|
|---|---|---|---|
|Qwen3.5-4B|2160|0.987|1.000|
|Falcon-H1-1.5B|2160|0.988|1.002|

奇异谱、token 分支、site 和 family 标注在 calibration-only Parquet/NPZ。这里是 exact raw-state 线性映射的解析几何；没有前瞻因果低秩压缩检验，故不称低秩功能机制。
