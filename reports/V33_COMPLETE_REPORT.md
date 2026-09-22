# V33 Complete Report

**Cross-Model Replication of REC–Conv Conditional Correction** — *Is Conv-Dominant Handoff + REC Residual Correction a General Hybrid-LM Mechanism?*

Independent model: `tiiuae/Falcon-H1-1.5B-Base`, revision `06d7330266253c20784f58b0a846dc09a9d12cee`, checkpoint hash `9acd2ef3e946e88a6e8cb14d38942f2ffbf6f52a5bc37312c1e0cc64574f1aeb`. All 24 layers expose exact persistent Mamba-2 REC2, short-Conv CONV2 and attention KV fields. The high-level field transaction is comparable to V32, but microscopic update algebra differs. Model choice, architecture map, 25/100/50/50 roles, ordinary token rule, six-probe response bundle and gates were frozen before model-2 causal outcomes.

|role|n|positive|median reduction|reduction LB|median align|align LB|correction fam|alignment fam|
|---|---|---|---|---|---|---|---|---|
|development|100|1.000|0.388|0.351|0.793|0.766|5|5|
|validation|50|1.000|0.413|0.300|0.811|0.725|5|5|
|independent final|50|1.000|0.363|0.312|0.776|0.736|5|5|

Conv-only/REC-only/joint validation donor relative L2 are 0.240/1.001/0.126; exact joint beats the descriptive best scalar in every validation row. Development-only fixed rank-32 rotation is beaten by true joint in 1.000 of validation rows. Matched REC2 beats the three frozen primary mismatch controls in 20/20 states; same/cross-state artificial controls carry off-manifold caveats. KV-only median relative error 0.984 on the 20-state contrast subset. Interaction ratios are 0.224/0.228/0.219.

Formal V33 A/B/C/D **pass**. E/F/G do not apply to the observed result. Development and validation passed the locked correction/alignment gates; only then was the 50-state model-2 final opened, and it confirmed. The sealed V32 model-1 final was not reopened. This supports cross-model replication in **the two tested hybrid LMs**, not universality.

Phase B was authorized. One-state-per-model kernel instrumentation audited exact transformed Qwen gates and true delta, and Falcon post-Conv factors and true dBx in 24/24 layers each. Bidirectional multi-state mediation was not executed; V33-H/I remain **not established**, not falsified. Applications remain unbenchmarked. H2 remains; H3, dynamic-state search, and autonomous controller remain unauthorized.

Parent `09a00a67e5a286a05e47730c43586bce94b8095a`; V33 base freeze `aa843d414c7ad5866a47beebc665db9d0379d2a622215b1465cdc4d6ae41b600`; final adjudication freeze `7d9dd78515dca769d3af7c4542537ee3e8077ea802d91e6670e53ef0d5343869`. See per-topic V33 reports and the machine integrity index. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.
