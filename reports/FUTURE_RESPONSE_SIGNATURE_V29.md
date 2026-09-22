# Future Response Signature — V29

Ten frozen states per role used four next-token probes selected from the **prefix distribution before either write**. The concatenated functional signature tests whether a channel tracks donor response across inputs, not merely a single `z`. Complete-cache signatures are exact; REC+Conv remains strongly donor-directed; KV remains weak. The probe-set choice limits generalization beyond these tokens.

| role | channel | A←B cosine / rel-L2 | B←A cosine / rel-L2 |
|---|---|---:|---:|
| development | REC | 0.277 / 0.969 | 0.236 / 0.982 |
| development | Conv | 0.966 / 0.258 | 0.940 / 0.345 |
| development | KV | 0.227 / 0.975 | 0.210 / 0.980 |
| development | REC+Conv | 0.988 / 0.152 | 0.989 / 0.147 |
| development | REC+Conv+KV | 1.000 / 0.000 | 1.000 / 0.000 |
| validation | REC | 0.277 / 0.975 | 0.268 / 0.974 |
| validation | Conv | 0.951 / 0.307 | 0.932 / 0.363 |
| validation | KV | 0.239 / 0.976 | 0.214 / 0.978 |
| validation | REC+Conv | 0.986 / 0.167 | 0.989 / 0.149 |
| validation | REC+Conv+KV | 1.000 / 0.000 | 1.000 / 0.000 |
