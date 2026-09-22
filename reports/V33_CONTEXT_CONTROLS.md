# V33 Matched and Mismatched REC2 Context Controls

Ten predeclared development plus ten validation targets (two per family per role). Matching Conv was held fixed. Correct matched REC2 beat wrong-token, same-family wrong-state and layer-shuffled REC2 in **20/20** targets each; the all-three success fraction is 1.000. Each of five families passes in development and validation.

|condition|median donor relative L2|matched better fraction|
|---|---:|---:|
|WRONG_TOKEN_REC2|0.289|1.000|
|SAME_FAMILY_WRONG_STATE_REC2|0.293|1.000|
|SHUFFLED_REC2|1.795|1.000|
|CROSS_FAMILY_WRONG_STATE_REC2|0.481|1.000|
|RANDOM_SAME_NORM_REC2|2.492|1.000|
|MATCHED_REC2|0.157|—|
|RECIPIENT_REC2 / Conv-only|0.252|—|

Cross-state, layer-shuffle and random controls can be off-manifold; wrong-token same-prefix is the cleanest context-specific contrast. Mapping was frozen after primary factorial but before these control responses. V33-D **PASS within the frozen 20-state confirmatory subset**, with that narrower timing acknowledged. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.
