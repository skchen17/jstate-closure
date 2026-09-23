# V37 标量增益与冻结旋转对照

|模型|阶段|解释|方向余弦|相对向量误差|相对幅度误差|
|---|---|---|---|---|---|
|Qwen3.5-4B|raw|native|1.000|0.000|0.000|
|Qwen3.5-4B|raw|gain|0.961|0.294|0.061|
|Qwen3.5-4B|raw|fixed_rotation|0.973|0.250|0.059|
|Qwen3.5-4B|mixer|native|1.000|0.000|0.000|
|Qwen3.5-4B|mixer|gain|0.959|0.302|0.072|
|Qwen3.5-4B|mixer|fixed_rotation|0.962|0.293|0.071|
|Falcon-H1-1.5B|raw|native|1.000|0.000|0.000|
|Falcon-H1-1.5B|raw|gain|0.994|0.171|0.088|
|Falcon-H1-1.5B|raw|fixed_rotation|0.994|0.164|0.089|
|Falcon-H1-1.5B|mixer|native|1.000|0.000|0.000|
|Falcon-H1-1.5B|mixer|gain|0.985|0.255|0.123|
|Falcon-H1-1.5B|mixer|fixed_rotation|0.984|0.254|0.122|

标量增益与 rank-16 固定正交旋转均只在 development 的局部状态差上拟合，按模型×层位×阶段×目标算子冻结；validation 不重拟合。比较原生预测、增益和旋转的方向/向量误差/幅度误差。若简单对照接近原生，不能宣称复杂算子几何是独特解释。
