# V33 Gain and Fixed-Rotation Falsification

The descriptive per-row best scalar of CONV2 is beaten by the true joint response in 1.000 of validation rows; median scalar donor error 0.239 versus true joint 0.126. Development-only global scalar α=0.988 has validation median error 0.240, true joint better in 1.000.

Development-only rank-32 Procrustes map, identity outside fitted subspace, has validation median error 0.234; true joint is better in 1.000. This excludes this fixed low-capacity linear rotation, not every nonlinear or state-dependent map. Scalar fit is descriptive oracle; no intervention data were used to retune the primary gate. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.
