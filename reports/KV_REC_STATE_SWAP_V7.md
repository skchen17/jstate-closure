# KV / Recurrent State Swap v7

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Mode: `causal attribution and corrective compression`
- Date: `2026-09-15`
- Verification Status: `repository-grounded; machine records and 10,000-resample CIs`
- Protocols: `persistent_channel_attribution_protocol_v7`, `persistent_channel_compression_corrective_v7_1`
- attribution freeze: `d2bc61ef18093d50adc9c0c02578c853b9b01103aea9acd797fe7f6755b09cf8`
- localization analysis freeze: `494071a74252419727ea2ab9b7fb5c4091831c19564f805c4d04dba8e2c2916a`
- corrective compression freeze: `a4c31a47f3e2ef3126f49f6988f38dd4a45882c0ff8c0b11328fa2bf5093de68`
- captured channel tensor SHA-256: `30c8abc4f3dee43594e382c49e74ff43baaee909e76d6c45a62bcd7d2c48d88b`


## Causal design and result

For every one of 33 held-out pairs, clean, KV-only,
REC+conv-only, and full caches consumed identical clean teacher tokens. The
pooled next-J results are:

| condition | next-J L2 | direction cosine | magnitude ratio | output JS | abs target log-odds delta |
| --- | --- | --- | --- | --- | --- |
| clean | 0.000000 [0.000000, 0.000000] | NA | 0.000000 [0.000000, 0.000000] | 0.00000000 [-0.00000000, 0.00000000] | 0.000000 [0.000000, 0.000000] |
| kv_only | 0.008886 [0.007704, 0.010408] | 0.241406 [0.196983, 0.288726] | 0.291590 [0.261220, 0.323641] | 0.00003206 [0.00001861, 0.00004750] | 0.022639 [0.017109, 0.028569] |
| recurrent_only | 0.030665 [0.027501, 0.034032] | 0.952030 [0.939131, 0.962745] | 0.974823 [0.956303, 0.990817] | 0.00018725 [0.00006236, 0.00039987] | 0.050222 [0.031540, 0.073656] |
| full | 0.031605 [0.028197, 0.035152] | 1.000000 [1.000000, 1.000000] | 1.000000 [1.000000, 1.000000] | 0.00017820 [0.00005556, 0.00038820] | 0.047598 [0.031143, 0.068244] |

Recurrent+conv reproduces cosine
0.952030 [0.939131, 0.962745] and magnitude
0.974823 [0.956303, 0.990817]; KV alone reproduces
cosine 0.241406 [0.196983, 0.288726] and magnitude
0.291590 [0.261220, 0.323641]. The non-additive vector
interaction is 0.305402 [0.276751, 0.334826]. Output-JS interaction is
-0.00004111 [-0.00007555, -0.00000974].

## Family heterogeneity

| family | n | KV cosine | REC cosine | interaction ratio |
| --- | --- | --- | --- | --- |
| boolean_logic | 12 | 0.160079 [0.097808, 0.221514] | 0.974468 [0.966103, 0.982015] | 0.268236 [0.228369, 0.313158] |
| modular_arithmetic | 6 | 0.299160 [0.201487, 0.421121] | 0.937964 [0.882581, 0.971595] | 0.242896 [0.220756, 0.258997] |
| short_graph_traversal | 6 | 0.357815 [0.261954, 0.457097] | 0.931972 [0.913917, 0.950841] | 0.345254 [0.277728, 0.406418] |
| simple_state_transition | 3 | 0.245652 [0.157987, 0.327130] | 0.949886 [0.933390, 0.963708] | 0.391103 [0.334985, 0.422575] |
| variable_binding | 6 | 0.227776 [0.160863, 0.275876] | 0.942350 [0.930526, 0.954811] | 0.359537 [0.298937, 0.414281] |

The recurrence result is directionally stable across all five families, but
family n ranges from 3 to
12; family comparisons are exploratory.
No task-decision flips occur often enough here to localize a family-specific
behavioral mechanism. The causal conclusion is strongest for next-J writes and
logit-distribution changes.
