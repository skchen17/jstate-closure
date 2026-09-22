# V33 Primary REC2×CONV2 Factorial

|role|n|positive|median reduction|reduction LB|median align|align LB|correction fam|alignment fam|
|---|---|---|---|---|---|---|---|---|
|development|100|1.000|0.388|0.351|0.793|0.766|5|5|
|validation|50|1.000|0.413|0.300|0.811|0.725|5|5|
|independent final|50|1.000|0.363|0.312|0.776|0.736|5|5|

All five native conditions were measured on each of 100 development, 50 validation and—after explicit opening—50 independent-final states, with six frozen probes each. Full tensors are in role-specific `factorial_*_v33.parquet`, `factorial_vectors_*_v33.npz` and `factorial_audit_*_v33.parquet`. Every primary hybrid retained recipient KV; writeback and per-condition field hashes were audited. Development and validation satisfy both frozen gates, and the independent final confirms direction without model/threshold adaptation. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.
