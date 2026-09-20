# V19 writeback order and quantization audit

Frozen diagnostic subset: 5 validation base states. A=snapshot Pq→a (primary construction); B=direct q→a; C=direct a→q; D=single composed write. B/C/D are not used for V19 primary claims.

| branch | median stack distance from A | median distance / ‖A‖ |
|---|---|---|
| A_snapshot_q_then_a | 0.0000 | 0.0000 |
| B_direct_q_then_a | 0.0000 | 0.0000 |
| C_direct_a_then_q | 0.1758 | 0.0030 |
| D_single_composed_write | 0.1691 | 0.0030 |

B exact persistent snapshot identity versus A: 1.0000. BF16 order and composition effects are quantified in `results/v19/processed/writeback_order_audit_v19.parquet`; the primary factorial always branches from exact P0/Pq snapshots.

Sign/scale diagnostic: 10 states; 800 observations for selected q at ± signs and α=0.5/1.0 across h1/h2/h4/h8; q reliability 1.0000. Median M/R by scale: {'0.5': 0.6318129897117615, '1.0': 0.8378927409648895}. Odd/even components are in `results/v19/processed/q_sign_odd_even_v19.parquet`; no linearity assumption is made.
