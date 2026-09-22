# V33 Conv-Dominant Handoff

|role|REC-only donor relative L2|CONV2-only donor relative L2|REC2+CONV2 relative L2|CONV/REC error ratio|
|---|---:|---:|---:|---:|
|development|0.997|0.242|0.135|0.244|
|validation|1.001|0.240|0.126|0.244|
|final|0.997|0.227|0.140|0.228|

All 5/5 families pass the prospective CONV2 dominance criterion in each role. This is an immediate one-token future readout test, not a claim that attention KV or REC2 is generally unimportant. V33-B **PASS**. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.
