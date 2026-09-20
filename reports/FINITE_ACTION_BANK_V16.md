# Reliable finite-action response bank — V16

Pre-response frozen split: train **100** states (20 per family), validation **50** states (10 per family), five families each; ID SHA256 train `d2c6cc991ebe5eb85a47e36895256ad5777141a9f334bacae3779c3136adbe4a`, validation `9dc52e5e18e2fbec634427fa91935ae192c445de2d80a4c765386e39e551aa43`. V15 development IDs were excluded from V16 validation. A newly generated independent final bank was **not opened** because the response validation gate did not pass. This is a validation bank, not independent confirmatory control evidence.

The 32 nested proposal directions draw from V13 autograd causal-weighted, architecture-balanced and random raw directions plus V15 train-selected high finite response and poor-autograd-subspace-residual directions. Direction indices and family labels are in the split freeze. Every action uses real joint REC/Conv/KV BF16 writeback and a state/direction-specific first-passing amplitude among [0.5, 1.0, 1.5, 2.0]. Both signs must meet state cosine ≥0.95, gain 0.8–1.2 and J effect ≥0.0068028033. Failure is retained as `ACTION_NOT_RELIABLY_ACTUATABLE`.

- Train rows: 7440; reliable 6613.
- Validation rows: 3720; reliable 3327.
- Overall reliable-action fraction: 0.891.
- Designs: {"dense": {"count": 240, "reliable": 240}, "pair": {"count": 480, "reliable": 480}, "scale": {"count": 720, "reliable": 662}, "single": {"count": 9600, "reliable": 8438}, "triple": {"count": 120, "reliable": 120}}.

Per-row requested alpha/coordinate, realized norm/cosine/gain, channel survival, five target responses and failure labels are in `/data/CSK/J-space-project/jstate-closure/results/v16/processed/finite_action_bank_v16.parquet` (SHA256 `680c01ed70c44a96bb46e8429eebee1e2ec3f6ebe5cecb6676ea114fd3025eef`). Raw per-state Parquet remains in `results/v16/raw/`, preserving the actual split and all failures. Calibration attempts before the final selected amplitude were not all retained as separate target-response rows; the selected/last tried amplitude is explicit.
