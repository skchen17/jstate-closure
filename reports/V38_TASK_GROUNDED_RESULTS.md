# V38 — Task Grounded Results

One-token answer diagnostic (invalid/coverage-fail states excluded):

|model|role|valid|margin positive|median margin Δ|R000 accuracy|R111 accuracy|
|---|---|---|---|---|---|---|
|Q|development|18|0.389|-0.125|0.833|0.778|
|Q|validation|9|0.111|-0.250|0.778|0.778|
|F|development|18|0.444|-0.016|0.111|0.111|
|F|validation|9|0.333|-0.016|0.222|0.222|

R111 does not consistently improve the externally defined correct-answer margin or accuracy over R000. Neither model shows a stable task-grounded effect. Probe/format and missing semantic contrasts further limit interpretation; V38-G is not established.
