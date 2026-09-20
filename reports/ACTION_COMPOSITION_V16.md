# Finite action composition and order — V16

Ten frozen states (first per family from train and validation) test two reliable actions. Same-state u→v versus v→u measures BF16 writeback order; +u→+v→−u→−v is an empirical finite loop, **not** a Lie bracket. Teacher-forced dynamic order reuses the same raw directions after one controlled token; proper tangent transport was not estimated.

- Measured states: 10.
- Median same-state J order L2: 0.001370.
- Median dynamic J order L2: 0.063462.
- Median finite-loop J residual relative to larger first-order J effect: 0.014.
- States above frozen static-order noncommutativity threshold: 7.

The threshold is UV-versus-VU J norm >1e−6 and relative to the larger first-order effect >0.01. The restricted tested-regime label is **NONCOMMUTATIVE_ACTION_COMPOSITION_PRESENT**. Loop residuals are reported separately and do not alone prove order dependence. State/output/semantic order effects are in `/data/CSK/J-space-project/jstate-closure/results/v16/processed/action_composition_v16.parquet`. Dynamic order effects mix writeback and token-transition dependence; they are not proof of a smooth noncommutative geometric bracket.
