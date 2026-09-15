# J-State Closure Final Report

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


## Current adjudication

The strongest warranted conclusion remains **H2 for the tested operational
measured-J state**. v6 established that same current measured-J plus a different
operational remainder causally changes the next within-forward measured-J write.
v7 now localizes the cross-token persistence to a **mixed, interacting KV and
Gated-DeltaNet recurrent/short-conv cache state**, with recurrent+conv dominant.
This rules out the claim that the observed effect is only a nonpersistent local
transient. It does not establish H3: no architecture-aligned compact state
passed conditional sufficiency, and no autonomous controller was trained.

## v7 decisive measurements

| condition | next-J L2 | direction cosine | magnitude ratio | output JS | abs target log-odds delta |
| --- | --- | --- | --- | --- | --- |
| clean | 0.000000 [0.000000, 0.000000] | NA | 0.000000 [0.000000, 0.000000] | 0.00000000 [-0.00000000, 0.00000000] | 0.000000 [0.000000, 0.000000] |
| kv_only | 0.008886 [0.007704, 0.010408] | 0.241406 [0.196983, 0.288726] | 0.291590 [0.261220, 0.323641] | 0.00003206 [0.00001861, 0.00004750] | 0.022639 [0.017109, 0.028569] |
| recurrent_only | 0.030665 [0.027501, 0.034032] | 0.952030 [0.939131, 0.962745] | 0.974823 [0.956303, 0.990817] | 0.00018725 [0.00006236, 0.00039987] | 0.050222 [0.031540, 0.073656] |
| full | 0.031605 [0.028197, 0.035152] | 1.000000 [1.000000, 1.000000] | 1.000000 [1.000000, 1.000000] | 0.00017820 [0.00005556, 0.00038820] | 0.047598 [0.031143, 0.068244] |

Corrected raw-channel cosine gain is 0.007552 [0.005535, 0.009790]; the
best compression closes 0.474772 [0.419485, 0.526364] of the direct
channel gap but leaves 0.525228 [0.473636, 0.580515] conditional
gain. Candidate sufficient state:
`False`.

## Scientific interpretation

- **Causal:** v6 same-J intervention; v7 KV/REC cache swaps and direct compressed-cache reconstructions.
- **Predictive:** held-out raw-channel ceiling and learned component ordering.
- **Practical magnitude:** REC+conv reproduces most next-J direction, but output JS is small and answer flips are rare.
- **Uncertainty:** only 33 held-out pairs; family cells have 3--12 items; compression rank is 32.

## Validity and fallacy scan

1. Predictive gains are not called causal; only state swaps support causal wording.
2. Finite 4096D profiles remain “measured-J,” not complete J-space.
3. Pooled and family-wise attribution are both reported.
4. Fit-half localization is separated from held-out attribution.
5. The endpoint archive defect is preserved and disclosed; corrected analysis has its own freeze.
6. Ridge alpha, split, bootstrap seed, and thresholds were not tuned after outcomes.
7. Nominal dimensions above empirical rank are not counted as independent evidence.
8. Failure to compress does not prove no compact representation exists.
9. Token-window equality is not generalized beyond final-token intervention scope.
10. Interaction prevents calling REC or KV independently sufficient.
11. No consciousness, true-thought, parameter-localization, or autonomous-controller claim is made.

## Next blocker

Collect a substantially larger independent architecture-state bank so 64--512
dimensions are identifiable, then fit channel-specific nonlinear causal
bottlenecks with explicit ordinary next-state, semantic, output, and conditional
residual endpoints. Only a candidate passing all frozen sufficiency gates should
authorize autonomous recurrent dynamics.

## Exact v7 commands

```bash
scripts/run_persistent_channels_v7.sh schema --run-suffix cache-schema
scripts/run_persistent_channels_v7.sh restore --run-suffix restore-full8
CUDA_VISIBLE_DEVICES=0 scripts/run_persistent_channels_v7.sh attribution --run-suffix attribution-full66
scripts/run_persistent_channels_v7.sh analyze --run-suffix attribution-analysis
CUDA_VISIBLE_DEVICES=0 scripts/run_arch_state_v7.sh localize --run-suffix localization-full66
scripts/run_localization_analysis_v7.sh --run-suffix analysis-corrected-r1
scripts/run_arch_compression_v7_corrective.sh freeze --run-suffix endpoint-fix-freeze
scripts/run_arch_compression_v7_corrective.sh ceiling --run-suffix corrected-endpoint-ceiling
CUDA_VISIBLE_DEVICES=0 scripts/run_arch_compression_v7_corrective.sh run --run-suffix corrected-compression-full
scripts/run_arch_compression_v7_corrective.sh analyze --run-suffix corrected-compression-analysis
scripts/build_report_v7.sh --run-suffix final-v7
```

## Provenance continuity

The v6 causal endpoint freeze remains `f62c55d21e5e220cd21ea81c19a28bf1d5249a60bdc35a08e3eb029dd61ec12e`. All v1--v6
frozen reports/results remain unmodified; `FINAL_REPORT.md` is the declared
cumulative-report exception.
