# V33 Post-Conv Factor Capture / Mediation

Qwen3.5: the actual cached convolution-update output and q/k/v consumed by the recurrent kernel were captured and hashed for 24 layers, rather than substituted with pre-convolution projections. Falcon-H1: the actual post-convolution x/B/C Mamba factors were captured in 24 layers. These are analogous post-Conv recurrence inputs but not identical q/k/v anatomy.

No writable post-Conv factor removal and reverse restoration was validated in both models. **Shared q/k/v mediation: NOT ESTABLISHED.** Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.
