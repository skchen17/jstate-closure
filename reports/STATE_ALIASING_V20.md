# V20 workspace-state aliasing

Across 150 validation P0/Pq pairs, boundary J is exactly equal: **True**. Median standardized oracle-coordinate distance=12.3157; median train-action fingerprint difference=0.8083; fraction exceeding frozen coordinate and response thresholds=1.0000. Operational `WORKSPACE_STATE_ALIASING` observed: **True**.

The same J necessarily gives the same deterministic J-only prediction within a pair, while the measured train-action fingerprint can differ. This is an operational finite-action aliasing result, not a complete POMDP or Markov-state proof. Pair file `results/v20/processed/operator_aliasing_pairs_v20.parquet` (`20fd325b427ab5e793e548758b93f39213caac62fd3ded9f72265678c35fc005`).
