# V36 上游上下文依赖

NATURAL_FORWARD 让 AA/BA/AB/BB 各有自然上游输入；FIXED_LOCAL_INPUT 将四格都置于 Conv-only 的同一个 mixer 输入。记录两种交互的逐 probe 范数与余弦，并以状态中位为单位。

|角色|模型|固定/自然交互范数中位比|局部下游收益恢复率|
|---|---|---|---|
|development|Qwen3.5-4B|0.185|0.030|
|development|Falcon-H1-1.5B|0.117|0.020|
|validation|Qwen3.5-4B|0.187|0.032|
|validation|Falcon-H1-1.5B|0.109|0.028|

固定输入交互显著缩小，是上游变化很重要的证据；但未在协议中冻结 V36-C 的单独充分门槛，因此只作描述性支持，不正式宣称上游路径单独解释了全部条件收益。
