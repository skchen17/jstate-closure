# Persistent Channel Attribution v7

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


## Outcome

The v6 same-measured-J causal effect is persistently represented: exact cache
restoration passed 8/8 trials with maximum
logit/hidden/J errors 0.000000/
0.000000/0.000000. The frozen
classification is **mixed_interacting**. Both KV and recurrent
channels are causally active; recurrent+conv dominates next-J direction, while
the pooled interaction ratio 0.305402 [0.276751, 0.334826] exceeds the
frozen additive limit 0.25.
The result is therefore neither a pure-KV nor a pure-recurrent state.

## Cache schema

At sequence length 55, the cache occupies
27.219 MiB. Full-attention layers are
`[3, 7, 11, 15, 19, 23, 27, 31]`;
the other layers are Gated DeltaNet layers.

| component | tensor count | bytes | MiB |
| --- | --- | --- | --- |
| conv_states | 24 | 1572864 | 1.500 |
| keys | 8 | 901120 | 0.859 |
| recurrent_states | 24 | 25165824 | 24.000 |
| values | 8 | 901120 | 0.859 |

## Paired 2×2 attribution (held-out n=33)

| condition | next-J L2 | direction cosine | magnitude ratio | output JS | abs target log-odds delta |
| --- | --- | --- | --- | --- | --- |
| clean | 0.000000 [0.000000, 0.000000] | NA | 0.000000 [0.000000, 0.000000] | 0.00000000 [-0.00000000, 0.00000000] | 0.000000 [0.000000, 0.000000] |
| kv_only | 0.008886 [0.007704, 0.010408] | 0.241406 [0.196983, 0.288726] | 0.291590 [0.261220, 0.323641] | 0.00003206 [0.00001861, 0.00004750] | 0.022639 [0.017109, 0.028569] |
| recurrent_only | 0.030665 [0.027501, 0.034032] | 0.952030 [0.939131, 0.962745] | 0.974823 [0.956303, 0.990817] | 0.00018725 [0.00006236, 0.00039987] | 0.050222 [0.031540, 0.073656] |
| full | 0.031605 [0.028197, 0.035152] | 1.000000 [1.000000, 1.000000] | 1.000000 [1.000000, 1.000000] | 0.00017820 [0.00005556, 0.00038820] | 0.047598 [0.031143, 0.068244] |

| family | n | KV cosine | REC cosine | interaction ratio |
| --- | --- | --- | --- | --- |
| boolean_logic | 12 | 0.160079 [0.097808, 0.221514] | 0.974468 [0.966103, 0.982015] | 0.268236 [0.228369, 0.313158] |
| modular_arithmetic | 6 | 0.299160 [0.201487, 0.421121] | 0.937964 [0.882581, 0.971595] | 0.242896 [0.220756, 0.258997] |
| short_graph_traversal | 6 | 0.357815 [0.261954, 0.457097] | 0.931972 [0.913917, 0.950841] | 0.345254 [0.277728, 0.406418] |
| simple_state_transition | 3 | 0.245652 [0.157987, 0.327130] | 0.949886 [0.933390, 0.963708] | 0.391103 [0.334985, 0.422575] |
| variable_binding | 6 | 0.227776 [0.160863, 0.275876] | 0.942350 [0.930526, 0.954811] | 0.359537 [0.298937, 0.414281] |

These are intervention-based effects from chimeric cache states under a shared
teacher-forced token sequence. They localize the persistence path; they do not
show that the cache channel is compact or sufficient.

## Endpoint serialization correction

The first auxiliary endpoint archive wrote the final `full` trajectory under
all condition names. Per-trial scalars and curves above were calculated before
that save and remain valid. The archive is preserved; corrective protocol v7.1
uses the frozen v6 step-1 vectors and a separate freeze.
