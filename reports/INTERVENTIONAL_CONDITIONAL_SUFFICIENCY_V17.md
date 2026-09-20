# V17 interventional conditional-sufficiency test

Primary desired test: `Y ⟂ P_t | (J_t,C_t,a_t)`. No compact `C` was selected because the prerequisite material raw-context ceiling gate failed. The executable **diagnostic** is therefore `C = ∅`, not a compact-state sufficiency claim.

For each reliable single-action coordinate/sign key, a J-based predictor was trained; target residuals were generated out of fold across training states. Thirty-two train-kernel raw features were residualized against J using the same training-state folds. A separately fitted linear ridge correction used only the two OOF residuals; the J base predictor and validation set stayed fixed. Joint REC+Conv+KV correction changed validation J relative L2 by `-0.0104` and stack by `-0.0005`; stack bootstrap [2.5%, median, 97.5%] = `[-0.019534563660787544, -0.000692084010808453, 0.02037226311169412]`. Corrected performance did not improve. This null/negative restricted correction cannot establish `Y ⟂ P | J,a`; the ceiling model itself is not near-full on every target/family.

Candidate compact-context conditional raw-gain curves: **not generated**, because no C passed the earlier gate. This is a stage-gated non-result, not zero conditional gain. The full seven-channel diagnostic is machine-readable in `results/v17/processed/conditional_raw_residual_v17.json`.
