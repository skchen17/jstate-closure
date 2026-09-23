# V38 — Eight Condition Rec Factorial

Every development (80/model) and validation (40/model) state has all eight REC conditions. Before R111, singles and pairs were sealed and both predictions hashed. Each condition uses exact native REC-state replacement only; Conv/KV were checked unchanged, and all later computations evolved natively. State-wise outputs are in `trajectory_*_v38.parquet` and local ignored `.npz` arrays; per-condition REC hashes are in the `proof_json` column.
