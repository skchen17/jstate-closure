# V35 全读出移植下留一组必要性

|角色|模型|条件|修正保真损失/高层收益|
|---|---|---|---|
|开发|Q|FULL_MINUS_Q1|0.176|
|开发|Q|FULL_MINUS_Q2|0.188|
|开发|Q|FULL_MINUS_Q3|0.412|
|开发|Q|FULL_MINUS_Q4|0.611|
|开发|F|FULL_MINUS_Q1|0.082|
|开发|F|FULL_MINUS_Q2|0.124|
|开发|F|FULL_MINUS_Q3|0.483|
|开发|F|FULL_MINUS_Q4|0.458|
|验证|Q|FULL_MINUS_Q1|0.175|
|验证|Q|FULL_MINUS_Q2|0.156|
|验证|Q|FULL_MINUS_Q3|0.419|
|验证|Q|FULL_MINUS_Q4|0.541|
|验证|F|FULL_MINUS_Q1|0.084|
|验证|F|FULL_MINUS_Q2|0.123|
|验证|F|FULL_MINUS_Q3|0.423|
|验证|F|FULL_MINUS_Q4|0.437|

从完整联合读出移植开始，仅把指定四分位恢复为 Conv-only 读出。损失为相对全移植的供体误差上升，报告连续值；没有预先冻结的单组必要性二值阈值，因此不把任一留一组结果单独宣称为必要模块。
