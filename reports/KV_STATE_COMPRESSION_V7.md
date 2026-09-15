# KV State Compression v7

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


## Localization

| mechanism | direction cosine | magnitude ratio | output JS |
| --- | --- | --- | --- |
| kv_layer_27 | 0.241406 [0.196983, 0.288726] | 0.291590 [0.261220, 0.323641] | 0.00004348 [0.00000472, 0.00009171] |
| kv_layer_31 | NA | 0.000000 [0.000000, 0.000000] | 0.00004979 [0.00001636, 0.00009137] |
| kv_layer_27_head_0 | 0.115582 [0.082700, 0.145644] | 0.195207 [0.177528, 0.213658] | 0.00002011 [0.00000256, 0.00005102] |
| kv_layer_27_head_1 | 0.090917 [0.048903, 0.128557] | 0.196680 [0.178773, 0.215513] | 0.00004350 [0.00000686, 0.00008983] |
| kv_layer_27_head_2 | 0.125333 [0.087215, 0.160886] | 0.205940 [0.188356, 0.223747] | 0.00004144 [0.00001086, 0.00008222] |
| kv_layer_27_head_3 | 0.215896 [0.169317, 0.267211] | 0.280211 [0.248804, 0.313810] | 0.00004590 [0.00001149, 0.00009085] |

Layer 27 is the only attention layer before the
main measured layer and carries the measured next-J effect. Layer 31 is after
that measurement, so its next-J direction is structurally zero while it can
still affect output logits. Fit-half head ranking is
`[3, 2, 0, 1]`; head 3 is largest on held-out data.
The 1/4/16/64-token windows are identical because the originating v6
intervention modified only the final cached token. That is a scope result, not
evidence that older KV positions are generally irrelevant.

## Compression verdict

KV was not compressed as an isolated sufficient state because its effect is
weaker and the frozen attribution is mixed-interacting. Joint mixed-channel
compression also failed conditional sufficiency, so no minimum KV dimension or
fixed memory-slot count is warranted.
