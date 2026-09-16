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

<!-- V8-RESULTS:START -->

## Protocol v8 large-sample persistent-state update

The frozen five-family data produced `1100` valid paired states, including `250` independent final-test pairs. Raw full-persistent swapping changed next measured-J by 0.0239 [0.0231, 0.0246] L2. Recurrent-matrix plus convolution state aligned with the full-persistent causal direction at 0.9492 [0.9459, 0.9523].

Compression status is `COMPLETED_SCREEN_CAUSAL_GATED` and the smallest fully authorized state is `None`. Controller authorization is `False`. Predictive screening is not counted as causal or conditional sufficiency; decoded-state intervention gates remain mandatory.

The strongest conclusion therefore remains **H2 for the tested operational measured-J state** unless and until one compressed persistent state passes predictive, conditional-residual, and causal-fidelity gates together. Dense measured-J is not relabeled as a compact state.

Evidence labels: R0–R7/factorial cache swaps are intervention-based causal evidence; compression regressions are held-out predictive evidence; confidence intervals quantify sampling uncertainty but do not establish state minimality.

### Required v8 questions

1. KV/recurrent attribution replication: R4 direction 0.9492 [0.9459, 0.9523] and R5 standalone next-J 0.0067 [0.0066, 0.0069].
2. Recurrent/conv dominance: R4 magnitude 0.9780 [0.9731, 0.9830] versus R5 0.2976 [0.2880, 0.3078].
3. L27/H3 independent contribution: R5 next-J is 0.0067 [0.0066, 0.0069]; this is a direct cache-swap effect.
4. REC×KV interaction: `{'atoms': ['rec_matrix_all', 'kv_l27_h3'], 'order': 2, 'output_js_interaction': {'confidence': 0.95, 'estimate': -9.607998995126698e-06, 'lower': -1.4051941503881617e-05, 'n_clusters': 250, 'n_observations': 250, 'n_resamples': 10000, 'upper': -5.34600337186961e-06}, 'vector_interaction_ratio_to_full': {'confidence': 0.95, 'estimate': 0.36527972982152995, 'lower': 0.3541827020333572, 'n_clusters': 250, 'n_observations': 250, 'n_resamples': 10000, 'upper': 0.37682445652088237}}`.
5. Smallest raw component set passing the frozen raw screen: `R7`.
6. Full persistent ceiling: R7 next-J 0.0239 [0.0231, 0.0246].
7. Compression outcome: `COMPLETED_SCREEN_CAUSAL_GATED`; best predictive candidate shared_family_residual/causal_bottleneck/512D: predictive gap 1.0142, conditional gain 0.0607.
8. Smallest sufficient dimension: `None`.
9. Conditional residual gain near zero: `False` under every pooled/family gate.
10. Decoded causal intervention fidelity passed: `False`.
11. Best universal versus family-specific predictive gap: `0.9649471210904913` versus `0.21504578517298215`.
12. Candidate sufficient persistent state obtained: `False`.
13. Autonomous recurrent dynamics authorized: `False`.

<!-- V8-RESULTS:END -->

<!-- V9-RESULTS:START -->

## Protocol v9 conditional-sufficiency dimension update

Reporting amendment 3: `3e616dec14672b5dbc367adaab0646778cd8f5ffe6ff7aa1d8933b7cc913e1ce`. After visual QA, the conditional curve is restricted to the comparable v9 512/576/599D results and the residual-axis label is corrected; frozen analysis values are unchanged.

Reporting amendment 2: `6a72d399808c48910745099891f0d784d963dc566976df121afe1ea35093be4d`. It adds endpoint labels, comparator documentation, and the executed report command; frozen analysis values are unchanged.

Reporting amendment: `978671aa348c2175a84711a90fd72bf13a509e3b1abca881719c308c8ca13752`. It corrects only the next-J endpoint label filter; frozen analysis values are unchanged.

The v9 rank audit found an effective training rank of `599` from 600 frozen training pairs. Consequently 768/1024/1536/2048D are not statistically identifiable in this dataset and are reported as `NA`, not as 599D aliases.

### Required v9 answers

1. Conditional residual gain from 512D upward: `[{'dimension': 512, 'conditional_residual_gain': -0.0006906941942870616, 'conditional_lower': -0.0014503446108661592, 'conditional_upper': 5.965475682169149e-05}, {'dimension': 576, 'conditional_residual_gain': -0.00033734907582402227, 'conditional_lower': -0.0006641988767310977, 'conditional_upper': -3.1477800663560636e-05}, {'dimension': 599, 'conditional_residual_gain': 9.831894189119338e-06, 'conditional_lower': -7.578643877059221e-06, 'conditional_upper': 2.7718728128820656e-05}]`.
2. A validated dimension elbow was not established beyond the 599D sample-rank ceiling.
3. Smallest candidate sufficient dimension: `None`.
4. Largest 512D architecture-residual source: `recurrent`; full standalone-plus-interaction breakdown is in `reports/RESIDUAL_INFORMATION_LOCALIZATION_V9.md`.
5. Predictive and causal dimension requirements cannot yet be equated: decoded causal fidelity remained gated.
6. One-/two-/four-token results are recorded; eight-token status is `not measured: frozen v8 capture ends at four tokens`.
7. The independent effect-enriched subset contains `100` final-test pairs; its conclusion remains observationally gated.
8. Primary-method semantic fidelity by dimension: `512D=0.6260, 576D=0.6260, 599D=0.6260`.
9. No validated compact sufficient persistent state has been obtained.
10. The current limitation is a combination of sample-rank/state-capacity identification and representation/objective insufficiency; the data do not support claiming intrinsic incompressibility.
11. Autonomous-controller authorization: `False`.

The strongest warranted conclusion remains H2 for the tested measured-J state. Persistent information is compressible predictively, but no compact state has passed predictive, conditional, semantic, multi-horizon, and decoded causal gates together.

Exact commands:

```bash
scripts/run_sufficiency_v9.sh freeze --run-suffix protocol-freeze
scripts/run_sufficiency_v9.sh analyze --run-suffix rank-aware-full
scripts/run_sufficiency_v9_reporting_amendment_1.sh freeze --run-suffix label-filter-freeze
scripts/run_sufficiency_v9_reporting_amendment_2.sh freeze --run-suffix final-presentation-freeze
scripts/run_sufficiency_v9_reporting_amendment_3.sh freeze --run-suffix figure-qa-freeze
scripts/run_sufficiency_v9_reporting_amendment_3.sh report --run-suffix final-v9
```

<!-- V9-RESULTS:END -->

<!-- V10-RESULTS:START -->

## Protocol v10 corrected causal-sufficiency update

Base freeze `cf1f153f10f2aa9d522beb44ee6c0b7cb2df2d0b2353db72bcfa5d18eb0d3dc5`; residual amendment `f5b2e0654f389740058dadb9bd17951d8c623fe7f9352b401f381e29cd2d9cf7`; candidate freeze `3e2368a3b21738f7c00e8ca56290c03d228e978634e31491087bf7507da8a235`; causal metadata amendment `cb0df76882a701452f3973782b44aaa0ea5a0984881d81465c5e1037ecdef3a5`; strict-interface freeze `4b319ebffda82f7bcd155df70bbd7d0490dab9bea7fc92e8c1387ecb453d48c8`.

### Required v10 answers

1. **Yes.** v9 residual localization had coordinate duplication and was not the corrected joint conditional estimand.
2. At 512D, corrected rich-block gains are recurrent `0.1600`, conv `0.1605`, KV `0.1309`, joint `0.1673`; the exact same-599D combined reference is `-0.000691`.
3. The smallest validation/all-family observationally sufficient dimension is **384D**.
4. Yes. 512D is a strong reference/upper candidate, not the minimum.
5. Observational semantic sufficiency passes relative to a modest full ceiling (~`0.626`); absolute probe quality is ceiling-limited. Decoded-causal semantic fidelity fails separately.
6. Yes. 384/448/512D decode to recurrent/conv/KV state; full-rank feature reconstruction error is `7.153e-07`.
7. No. Compact decoded interventions do not reproduce all teacher raw-state effects under frozen gates.
8. No. Direction/output/semantic fidelity degrades through h2/h4/h8; h16 confirms continued instability.
9. No rescue: every effect-enriched candidate/horizon gate fails.
10. **No candidate causal sufficient persistent state was obtained.**
11. Main failures are decoder/interface semantic loss, block/channel interaction not captured by the combined ceiling, and long-horizon instability. Dimension alone and task heterogeneity are not sufficient explanations.
12. **Autonomous controller training is not authorized.**

H2/H3 adjudication: **H2 remains the strongest supported account; H3 is not established.**

### Exact commands

```bash
scripts/run_causal_sufficiency_v10.sh freeze --run-suffix protocol-freeze
scripts/run_causal_sufficiency_v10.sh audit --run-suffix corrected-audit
scripts/run_causal_sufficiency_v10_residual_amendment.sh freeze --run-suffix estimand-freeze
scripts/run_causal_sufficiency_v10_residual_amendment.sh audit --run-suffix frozen-base-audit
scripts/run_causal_sufficiency_v10.sh freeze-candidates --run-suffix candidate-freeze
scripts/run_causal_sufficiency_v10.sh prepare-decoder --device 0 --run-suffix decoder
scripts/run_causal_sufficiency_v10_causal_amendment.sh freeze --run-suffix metadata-freeze
HF_HOME=/data/CSK/J-space-project/.hf-cache scripts/run_causal_sufficiency_v10_causal_amendment.sh causal --device 0 --run-suffix causal-confirmatory-final
scripts/run_strict_interface_v10.sh freeze --run-suffix strict-freeze
scripts/run_strict_interface_v10.sh audit --run-suffix strict-three-way
scripts/run_causal_sufficiency_v10.sh report --run-suffix final-v10
scripts/finalize_reports_v10.sh --run-suffix amendment-aware-final
```

<!-- V10-RESULTS:END -->
