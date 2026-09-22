# V33 Transformed-Gate Instrumentation / Mediation

Qwen3.5: raw a/b projections were independently hooked and recomputed transformed g/beta matched the actual recurrent-kernel inputs exactly in 24/24 linear-attention layers for one future probe. Falcon-H1: transformed dt/decay were reconstructed from the actual in-projection and post-Conv factors, with exact state-update reconstruction in 24/24 layers. These are architecture-specific gate/decay operations, not identical algebra.

No gate-specific removal/reverse transplant across development and validation has been executed. **Gate mediation: NOT ESTABLISHED.** Raw projection alone was not used as a mediation claim. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.
