# Recurrent State Compression v7

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


## Localization before compression

| mechanism | direction cosine | magnitude ratio | output JS |
| --- | --- | --- | --- |
| recurrent_matrix_all | 0.340248 [0.273474, 0.401103] | 0.583368 [0.530710, 0.633975] | 0.00004762 [0.00001659, 0.00008738] |
| conv_all | 0.778085 [0.742550, 0.813104] | 0.944966 [0.913642, 0.977245] | 0.00053216 [0.00014954, 0.00108877] |
| recurrent_both_all | 0.952030 [0.939131, 0.962745] | 0.974823 [0.956303, 0.990817] | 0.00059856 [0.00014718, 0.00138167] |
| rec_layer_30_both | 0.610592 [0.573689, 0.649896] | 0.592266 [0.559751, 0.628418] | 0.00026834 [0.00008980, 0.00055536] |
| rec_layer_28_both | 0.417195 [0.373233, 0.463113] | 0.406082 [0.371856, 0.441502] | 0.00002740 [0.00000648, 0.00006030] |
| rec_layer_29_both | 0.330604 [0.262192, 0.395115] | 0.378250 [0.336329, 0.424614] | 0.00007837 [0.00003108, 0.00013579] |
| rec_layer_24_both | 0.361252 [0.318065, 0.402067] | 0.424748 [0.385071, 0.465917] | 0.00006030 [0.00001731, 0.00011520] |
| rec_layer_25_both | 0.270458 [0.225217, 0.315127] | 0.345995 [0.317716, 0.375134] | 0.00004976 [0.00001724, 0.00009104] |
| rec_layer_26_both | 0.262288 [0.220680, 0.303412] | 0.314417 [0.286556, 0.343437] | 0.00002733 [0.00000307, 0.00006125] |

Short-conv alone is the main recurrent-family carrier by direction and
magnitude, but matrix+conv jointly comes closest to the full mixed effect.
Layer ranking, selected only on the fit half, is
`[30, 28, 29, 24, 25, 26]`; layer 30 is the largest
single recurrent layer on held-out data. Because the global classification is
mixed-interacting, compression was run on the joint causal support rather than
pretending REC is independent of KV.

## Compression verdict

The effective sample rank is 32.
No candidate passed all frozen predictive, causal, direction, and conditional
sufficiency criteria (`nominal passing=0`).
Consequently there is no validated recurrent compact state and no justified
minimum REC dimension. Nominal 64--512 settings are rank-limited to 32 and do
not add independently observed degrees of freedom.
