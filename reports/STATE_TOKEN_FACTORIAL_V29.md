# State × Token Factorial — V29

Each frozen pair evaluates `(state A/B) × (token A/B)` with one shared next token. Exact state and token hashes, write-contrast geometry, response interaction norm, same-state REC+Conv reference, and wrong-state delta injection are in `state_token_factorial_*_v29.parquet`. This is a 10-pair-per-role diagnostic, not a global coordinate system proof.

| role | pairs | contrast cosine | cross-state delta→native cosine | cross-state delta relative L2 |
|---|---:|---:|---:|---:|
| development | 10 | 0.937 | 0.922 | 0.418 |
| validation | 10 | 0.958 | 0.948 | 0.347 |
