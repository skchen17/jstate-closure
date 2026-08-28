# Token-time closure report

## Status

GATED / NOT EXECUTED. Fresh official-compatible pass@10 was 0.775862 for multihop (58 items) and 0.716797 for order of operations (256 items). The item-clustered MRR advantage was 0.135202, 95% CI [0.109490, 0.161532]. At frozen layer 24, the intended-answer log-odds effect was 3.064270, 95% CI [1.978771, 4.221590], above the 0.0001 null envelope.

Independent closure-layer calibration then tested layers 23, 24, 25, 26, 27, 28, 29. It obtained 0/1400 strictly valid clamp trials; the best layer-level valid rate was 0.000, below the frozen 0.80 requirement. All candidate layers therefore failed only the clamp-valid-rate criterion, and the eligible set is empty.

T1, T2, and T3 macrostate construction and autonomous feedback code are implemented
and unit-tested, but no teacher traces, token-time predictors, recurrent memory models,
or autonomous rollouts were trained. No future teacher token is consumed by the rollout
interface. There is therefore no result about token-time closure, short-memory sufficiency,
controller size, procedural generalization, or intervention fidelity.
