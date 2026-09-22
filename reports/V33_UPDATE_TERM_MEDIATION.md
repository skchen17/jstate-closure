# V33 True Recurrent-Update Term Capture / Mediation

Qwen3.5: exact one-token torch kernel recurrence was replayed from captured consumed q/k/v, transformed g/beta and initial state; `delta` and rank-one write matched outgoing kernel state in 24/24 layers. Falcon-H1: post-Conv x/B plus transformed dt/decay reconstructed the actual dBx update and outgoing state argument bitwise in 24/24 layers. Per-layer hashes are in `phase_b_instrumentation_v33.json`.

The update terms are genuinely consumed/reconstructed, not inferred from raw projection only. However neither update term was intercepted bidirectionally to test ≥50% benefit removal/restoration. **Update-term mediation: NOT ESTABLISHED.** Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.
