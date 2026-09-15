# Architecture-Aligned Compact State v7

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Mode: `causal attribution and corrective compression`
- Date: `2026-09-15`
- Verification Status: `repository-grounded; machine records and 10,000-resample CIs`
- Protocols: `persistent_channel_attribution_protocol_v7`, `persistent_channel_compression_corrective_v7_1`
- attribution freeze: `d2bc61ef18093d50adc9c0c02578c853b9b01103aea9acd797fe7f6755b09cf8`
- localization analysis freeze: `494071a74252419727ea2ab9b7fb5c4091831c19564f805c4d04dba8e2c2916a`
- corrective compression freeze: `a4c31a47f3e2ef3126f49f6988f38dd4a45882c0ff8c0b11328fa2bf5093de68`
- captured channel tensor SHA-256: `30c8abc4f3dee43594e382c49e74ff43baaee909e76d6c45a62bcd7d2c48d88b`


## Adjudication

The persistent effect is real and mixed-interacting, but no candidate
sufficient persistent state was found. The corrected raw channel gives a
statistically positive yet practically small conditional predictive increment.
Compression then preserves a substantial causal direction but leaves about
half of the raw-channel effect recoverable from the omitted channel state.

## Raw-channel ceiling

| model | delta cosine | delta RMSE |
| --- | --- | --- |
| J only | 0.299040 [0.134964, 0.452359] | 0.001043 [0.000949, 0.001141] |
| J + mixed raw channel | 0.306591 [0.143751, 0.458990] | 0.001039 [0.000946, 0.001137] |
| paired gain/improvement | 0.007552 [0.005535, 0.009790] | 0.000004 [0.000003, 0.000004] |

Authorization: `True`; fit/test counts
33/33; effective rank
32. The predictive endpoint is held-out
intervention-delta prediction. The separate direct cache-swap attribution is
the causal ceiling; these evidence types are not conflated.

## Compression Pareto

| method | nominal D | effective D | predictive gap | causal gap | residual gain | direction cosine |
| --- | --- | --- | --- | --- | --- | --- |
| causal_bottleneck | 16 | 16 | 0.448098 [0.398360, 0.496550] | 0.758616 [0.726168, 0.790013] | 0.551902 [0.503450, 0.601640] | 0.837411 [0.807752, 0.865522] |
| causal_bottleneck | 32 | 32 | 0.474772 [0.419485, 0.526364] | 0.769515 [0.733485, 0.802872] | 0.525228 [0.473636, 0.580515] | 0.847820 [0.813759, 0.878286] |
| nonlinear_encoder | 16 | 16 | 0.182246 [0.099179, 0.257750] | 0.394962 [0.306404, 0.474745] | 0.817754 [0.742250, 0.900821] | 0.585781 [0.476446, 0.679521] |
| nonlinear_encoder | 32 | 32 | 0.474772 [0.419485, 0.526364] | 0.769515 [0.733485, 0.802872] | 0.525228 [0.473636, 0.580515] | 0.847820 [0.813759, 0.878286] |
| pca | 16 | 16 | 0.446455 [0.397384, 0.494324] | 0.753303 [0.720300, 0.785842] | 0.553545 [0.505676, 0.602616] | 0.835102 [0.804927, 0.863273] |
| pca | 32 | 32 | 0.474772 [0.419485, 0.526364] | 0.769515 [0.733485, 0.802872] | 0.525228 [0.473636, 0.580515] | 0.847820 [0.813759, 0.878286] |
| predictive_bottleneck | 16 | 16 | 0.376648 [0.322862, 0.428565] | 0.667589 [0.625182, 0.707809] | 0.623352 [0.571435, 0.677138] | 0.784570 [0.746311, 0.820335] |
| predictive_bottleneck | 32 | 32 | 0.474772 [0.419485, 0.526364] | 0.769515 [0.733485, 0.802872] | 0.525228 [0.473636, 0.580515] | 0.847820 [0.813759, 0.878286] |

The largest observed gap-closed estimate is
0.474772 [0.419485, 0.526364] for `causal_bottleneck` at nominal
32D/effective 32D. Its causal gap is
0.769515 [0.733485, 0.802872], direction cosine
0.847820 [0.813759, 0.878286], and conditional residual gain
0.525228 [0.473636, 0.580515]. Passing candidates:
0 independently
identified (0 nominal).

## Twelve required answers

1. **Can persistent state explain v6?** Yes. Both cache branches reproduce nonzero next-J/logit effects.
2. **Where is it?** Mixed KV and recurrent+conv with material interaction, not a purely transient local effect.
3. **Matrix or conv?** Short-conv is stronger alone; matrix+conv is strongest jointly.
4. **Which KV support?** Layer 27, especially head 3, for next measured-J; layer 31 is output-only relative to this endpoint. Token-window equality follows final-token intervention scope.
5. **Raw improvement over J-only?** Cosine gain 0.007552 [0.005535, 0.009790]; RMSE improvement 0.000004 [0.000003, 0.000004].
6. **Compressible?** Partially, but not to the frozen sufficiency criteria.
7. **Minimum effective dimension?** None established; even full empirical rank 32 fails.
8. **Predictive gap closed?** Best point estimate 0.474772 with CI 0.474772 [0.419485, 0.526364].
9. **Causal gap closed?** Best-candidate value 0.769515 [0.733485, 0.802872]; below the frozen lower-bound criterion.
10. **Residual information after compact state?** Yes: conditional residual gain 0.525228 [0.473636, 0.580515], far above the maximum 0.02.
11. **Candidate sufficient persistent state?** `False`.
12. **Ready for autonomous controller?** `False`. The protocol correctly stops before generic recurrent dynamics training.

## Evidence and limitations

- Cache swaps are causal interventions; raw prediction and bottleneck fitting are predictive/associational.
- The split has only 33 fit and 33 test pairs, with family n=3--12.
- Empirical rank 32 makes nominal 64--512 curves non-identifying.
- The raw predictor's positive increment is small despite tight paired CIs.
- Compression is of intervention deltas, not a complete natural-state model.
- The fixed-token continuation has no independent task-semantic decoder;
  task-decision evidence is limited to token and logit changes.
- No result localizes parameters, extracts “true thoughts,” or concerns consciousness.
