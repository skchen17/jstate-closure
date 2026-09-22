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

<!-- V11_START -->
## V11 — Architecture-resolved causal geometry

V11 used a new 50-pair independent confirmatory bank (10/family), disjoint from the 50 v10
development pairs. Development split hash:
`882c53041d9806ddde1915542e132dd14e43db1d146f9de1e53fffd999e4d635`; confirmatory split hash:
`b22941ec58285ad1cbed6f541d9cfe4ab1cee36983faf81dc16986f3f2851df3`; confirmatory program hash:
`db79342cf71638baa9e71b873937ce6931377c65dd314a400bcdf057d343626f`.

Frozen gates: `{"direction_cosine_minimum": 0.8, "magnitude_ratio_maximum": 1.2, "magnitude_ratio_minimum": 0.8, "output_direction_minimum": 0.8, "semantic_delta_agreement_minimum": 0.8, "task_decision_sign_minimum": 0.8}`.
Freeze digests: `{"base": "91cca911c33e1c5829d978fa307a54cfa7093118d039de35785a2fc388986ff3", "confirmatory": "753b48824a27a9b5934b774ba0594dc26d98a4d031c6a8c183b065c55470ca9f", "prepared": "58a31c7022b99b035b915054c64fe1b4e345b886838f75ea4d6ac401fb8675b8", "stage2": "d040b22427c48fcec1b4564b2fb14aa8d742b70047e11c1ce655a707e1308135"}`.

### Channel-wise hybrid interventions

| method/condition | h | direction | magnitude | semantic | output | sign | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| decoded_all | 1 | 0.923 | 0.977 | 0.614 | 0.896 | 0.820 | FAIL |
| decoded_all | 2 | 0.823 | 0.980 | 0.352 | 0.795 | 0.740 | FAIL |
| decoded_all | 4 | 0.688 | 0.987 | 0.206 | 0.660 | 0.696 | FAIL |
| decoded_all | 8 | 0.532 | 0.998 | 0.086 | 0.528 | 0.660 | FAIL |
| decoded_conv | 1 | 0.934 | 0.984 | 0.672 | 0.904 | 0.900 | FAIL |
| decoded_conv | 2 | 0.837 | 0.998 | 0.366 | 0.806 | 0.760 | FAIL |
| decoded_conv | 4 | 0.705 | 0.989 | 0.242 | 0.676 | 0.761 | FAIL |
| decoded_conv | 8 | 0.544 | 1.010 | 0.110 | 0.512 | 0.740 | FAIL |
| decoded_conv_kv | 1 | 0.932 | 0.978 | 0.656 | 0.907 | 0.860 | FAIL |
| decoded_conv_kv | 2 | 0.840 | 0.998 | 0.384 | 0.805 | 0.740 | FAIL |
| decoded_conv_kv | 4 | 0.703 | 0.991 | 0.228 | 0.671 | 0.761 | FAIL |
| decoded_conv_kv | 8 | 0.554 | 1.011 | 0.132 | 0.544 | 0.680 | FAIL |
| decoded_kv | 1 | 0.970 | 0.995 | 0.760 | 0.945 | 0.820 | FAIL |
| decoded_kv | 2 | 0.911 | 0.997 | 0.500 | 0.874 | 0.800 | FAIL |
| decoded_kv | 4 | 0.861 | 0.989 | 0.406 | 0.807 | 0.783 | FAIL |
| decoded_kv | 8 | 0.770 | 1.010 | 0.266 | 0.660 | 0.740 | FAIL |
| decoded_rec | 1 | 0.942 | 0.997 | 0.674 | 0.913 | 0.940 | FAIL |
| decoded_rec | 2 | 0.833 | 0.995 | 0.382 | 0.818 | 0.740 | FAIL |
| decoded_rec | 4 | 0.697 | 0.993 | 0.222 | 0.628 | 0.804 | FAIL |
| decoded_rec | 8 | 0.530 | 1.003 | 0.096 | 0.527 | 0.700 | FAIL |
| decoded_rec_conv | 1 | 0.923 | 0.979 | 0.624 | 0.900 | 0.860 | FAIL |
| decoded_rec_conv | 2 | 0.826 | 0.991 | 0.352 | 0.802 | 0.760 | FAIL |
| decoded_rec_conv | 4 | 0.691 | 0.987 | 0.212 | 0.652 | 0.565 | FAIL |
| decoded_rec_conv | 8 | 0.529 | 1.004 | 0.072 | 0.519 | 0.620 | FAIL |
| decoded_rec_kv | 1 | 0.941 | 1.000 | 0.682 | 0.914 | 0.940 | FAIL |
| decoded_rec_kv | 2 | 0.832 | 0.998 | 0.404 | 0.814 | 0.800 | FAIL |
| decoded_rec_kv | 4 | 0.697 | 0.995 | 0.212 | 0.655 | 0.696 | FAIL |
| decoded_rec_kv | 8 | 0.527 | 1.006 | 0.080 | 0.517 | 0.720 | FAIL |
| teacher_reference | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| teacher_reference | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| teacher_reference | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| teacher_reference | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |

### Oracle low-rank sweep

| method/condition | h | direction | magnitude | semantic | output | sign | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| oracle_factorized_pca_d128 | 1 | 0.182 | 0.332 | 0.048 | 0.225 | 0.540 | FAIL |
| oracle_factorized_pca_d128 | 2 | 0.300 | 0.564 | 0.044 | 0.328 | 0.640 | FAIL |
| oracle_factorized_pca_d128 | 4 | 0.389 | 0.775 | 0.062 | 0.384 | 0.609 | FAIL |
| oracle_factorized_pca_d128 | 8 | 0.483 | 0.975 | 0.086 | 0.459 | 0.720 | FAIL |
| oracle_factorized_pca_d128 | 16 | 0.465 | 0.940 | 0.066 | 0.478 | 0.596 | FAIL |
| oracle_factorized_pca_d192 | 1 | 0.184 | 0.328 | 0.040 | 0.221 | 0.620 | FAIL |
| oracle_factorized_pca_d192 | 2 | 0.312 | 0.567 | 0.040 | 0.308 | 0.680 | FAIL |
| oracle_factorized_pca_d192 | 4 | 0.400 | 0.770 | 0.062 | 0.419 | 0.761 | FAIL |
| oracle_factorized_pca_d192 | 8 | 0.480 | 0.976 | 0.074 | 0.453 | 0.760 | FAIL |
| oracle_factorized_pca_d192 | 16 | 0.468 | 0.947 | 0.086 | 0.438 | 0.681 | FAIL |
| oracle_factorized_pca_d256 | 1 | 0.173 | 0.342 | 0.064 | 0.237 | 0.560 | FAIL |
| oracle_factorized_pca_d256 | 2 | 0.308 | 0.571 | 0.078 | 0.355 | 0.540 | FAIL |
| oracle_factorized_pca_d256 | 4 | 0.407 | 0.780 | 0.072 | 0.376 | 0.543 | FAIL |
| oracle_factorized_pca_d256 | 8 | 0.487 | 0.979 | 0.090 | 0.485 | 0.760 | FAIL |
| oracle_factorized_pca_d256 | 16 | 0.461 | 0.947 | 0.070 | 0.444 | 0.553 | FAIL |
| oracle_factorized_pca_d320 | 1 | 0.211 | 0.342 | 0.054 | 0.259 | 0.500 | FAIL |
| oracle_factorized_pca_d320 | 2 | 0.315 | 0.574 | 0.032 | 0.352 | 0.640 | FAIL |
| oracle_factorized_pca_d320 | 4 | 0.401 | 0.773 | 0.058 | 0.409 | 0.457 | FAIL |
| oracle_factorized_pca_d320 | 8 | 0.469 | 0.972 | 0.062 | 0.501 | 0.780 | FAIL |
| oracle_factorized_pca_d320 | 16 | 0.465 | 0.939 | 0.090 | 0.493 | 0.660 | FAIL |
| oracle_factorized_pca_d384 | 1 | 0.227 | 0.345 | 0.082 | 0.259 | 0.580 | FAIL |
| oracle_factorized_pca_d384 | 2 | 0.331 | 0.578 | 0.058 | 0.348 | 0.700 | FAIL |
| oracle_factorized_pca_d384 | 4 | 0.414 | 0.787 | 0.054 | 0.469 | 0.609 | FAIL |
| oracle_factorized_pca_d384 | 8 | 0.481 | 0.983 | 0.074 | 0.510 | 0.700 | FAIL |
| oracle_factorized_pca_d384 | 16 | 0.464 | 0.945 | 0.112 | 0.479 | 0.638 | FAIL |
| oracle_factorized_pca_d448 | 1 | 0.246 | 0.366 | 0.074 | 0.276 | 0.620 | FAIL |
| oracle_factorized_pca_d448 | 2 | 0.360 | 0.608 | 0.068 | 0.389 | 0.620 | FAIL |
| oracle_factorized_pca_d448 | 4 | 0.438 | 0.798 | 0.078 | 0.403 | 0.587 | FAIL |
| oracle_factorized_pca_d448 | 8 | 0.483 | 0.967 | 0.074 | 0.471 | 0.600 | FAIL |
| oracle_factorized_pca_d448 | 16 | 0.482 | 0.932 | 0.096 | 0.472 | 0.574 | FAIL |
| oracle_factorized_pca_d512 | 1 | 0.346 | 0.409 | 0.102 | 0.293 | 0.580 | FAIL |
| oracle_factorized_pca_d512 | 2 | 0.420 | 0.644 | 0.082 | 0.407 | 0.500 | FAIL |
| oracle_factorized_pca_d512 | 4 | 0.466 | 0.819 | 0.068 | 0.445 | 0.565 | FAIL |
| oracle_factorized_pca_d512 | 8 | 0.486 | 0.979 | 0.094 | 0.495 | 0.740 | FAIL |
| oracle_factorized_pca_d512 | 16 | 0.485 | 0.957 | 0.078 | 0.478 | 0.596 | FAIL |
| oracle_factorized_pca_d64 | 1 | 0.172 | 0.332 | 0.070 | 0.251 | 0.520 | FAIL |
| oracle_factorized_pca_d64 | 2 | 0.307 | 0.566 | 0.046 | 0.295 | 0.620 | FAIL |
| oracle_factorized_pca_d64 | 4 | 0.392 | 0.786 | 0.056 | 0.416 | 0.674 | FAIL |
| oracle_factorized_pca_d64 | 8 | 0.488 | 0.973 | 0.100 | 0.490 | 0.720 | FAIL |
| oracle_factorized_pca_d64 | 16 | 0.466 | 0.939 | 0.082 | 0.468 | 0.574 | FAIL |
| oracle_joint_pca_d128 | 1 | 0.215 | 0.337 | 0.076 | 0.268 | 0.640 | FAIL |
| oracle_joint_pca_d128 | 2 | 0.321 | 0.569 | 0.030 | 0.339 | 0.700 | FAIL |
| oracle_joint_pca_d128 | 4 | 0.407 | 0.778 | 0.060 | 0.421 | 0.674 | FAIL |
| oracle_joint_pca_d128 | 8 | 0.475 | 0.982 | 0.092 | 0.492 | 0.800 | FAIL |
| oracle_joint_pca_d128 | 16 | 0.475 | 0.938 | 0.068 | 0.463 | 0.702 | FAIL |
| oracle_joint_pca_d192 | 1 | 0.303 | 0.353 | 0.102 | 0.291 | 0.620 | FAIL |
| oracle_joint_pca_d192 | 2 | 0.368 | 0.584 | 0.040 | 0.376 | 0.580 | FAIL |
| oracle_joint_pca_d192 | 4 | 0.424 | 0.783 | 0.058 | 0.433 | 0.587 | FAIL |
| oracle_joint_pca_d192 | 8 | 0.474 | 0.972 | 0.092 | 0.486 | 0.640 | FAIL |
| oracle_joint_pca_d192 | 16 | 0.476 | 0.947 | 0.084 | 0.452 | 0.638 | FAIL |
| oracle_joint_pca_d256 | 1 | 0.410 | 0.388 | 0.170 | 0.427 | 0.580 | FAIL |
| oracle_joint_pca_d256 | 2 | 0.422 | 0.607 | 0.100 | 0.440 | 0.620 | FAIL |
| oracle_joint_pca_d256 | 4 | 0.448 | 0.793 | 0.076 | 0.478 | 0.630 | FAIL |
| oracle_joint_pca_d256 | 8 | 0.485 | 0.973 | 0.084 | 0.495 | 0.660 | FAIL |
| oracle_joint_pca_d256 | 16 | 0.481 | 0.944 | 0.086 | 0.472 | 0.723 | FAIL |
| oracle_joint_pca_d320 | 1 | 0.717 | 0.621 | 0.362 | 0.662 | 0.700 | FAIL |
| oracle_joint_pca_d320 | 2 | 0.617 | 0.725 | 0.178 | 0.559 | 0.700 | FAIL |
| oracle_joint_pca_d320 | 4 | 0.545 | 0.855 | 0.116 | 0.503 | 0.739 | FAIL |
| oracle_joint_pca_d320 | 8 | 0.497 | 0.985 | 0.102 | 0.493 | 0.660 | FAIL |
| oracle_joint_pca_d320 | 16 | 0.502 | 0.968 | 0.092 | 0.457 | 0.660 | FAIL |
| oracle_joint_pca_d384 | 1 | 0.872 | 0.849 | 0.536 | 0.831 | 0.800 | FAIL |
| oracle_joint_pca_d384 | 2 | 0.768 | 0.872 | 0.300 | 0.736 | 0.720 | FAIL |
| oracle_joint_pca_d384 | 4 | 0.645 | 0.938 | 0.186 | 0.607 | 0.804 | FAIL |
| oracle_joint_pca_d384 | 8 | 0.513 | 0.993 | 0.118 | 0.508 | 0.640 | FAIL |
| oracle_joint_pca_d384 | 16 | 0.513 | 0.984 | 0.092 | 0.503 | 0.596 | FAIL |
| oracle_joint_pca_d448 | 1 | 0.915 | 0.939 | 0.636 | 0.890 | 0.820 | FAIL |
| oracle_joint_pca_d448 | 2 | 0.811 | 0.951 | 0.332 | 0.771 | 0.700 | FAIL |
| oracle_joint_pca_d448 | 4 | 0.674 | 0.961 | 0.184 | 0.652 | 0.761 | FAIL |
| oracle_joint_pca_d448 | 8 | 0.523 | 1.010 | 0.098 | 0.510 | 0.660 | FAIL |
| oracle_joint_pca_d448 | 16 | 0.532 | 0.984 | 0.114 | 0.518 | 0.702 | FAIL |
| oracle_joint_pca_d512 | 1 | 0.921 | 0.963 | 0.618 | 0.890 | 0.820 | FAIL |
| oracle_joint_pca_d512 | 2 | 0.816 | 0.964 | 0.326 | 0.779 | 0.700 | FAIL |
| oracle_joint_pca_d512 | 4 | 0.684 | 0.973 | 0.240 | 0.642 | 0.739 | FAIL |
| oracle_joint_pca_d512 | 8 | 0.523 | 0.997 | 0.118 | 0.514 | 0.720 | FAIL |
| oracle_joint_pca_d512 | 16 | 0.534 | 0.994 | 0.134 | 0.494 | 0.660 | FAIL |
| oracle_joint_pca_d64 | 1 | 0.177 | 0.327 | 0.088 | 0.243 | 0.460 | FAIL |
| oracle_joint_pca_d64 | 2 | 0.309 | 0.574 | 0.056 | 0.346 | 0.440 | FAIL |
| oracle_joint_pca_d64 | 4 | 0.396 | 0.772 | 0.062 | 0.392 | 0.717 | FAIL |
| oracle_joint_pca_d64 | 8 | 0.479 | 0.975 | 0.092 | 0.481 | 0.640 | FAIL |
| oracle_joint_pca_d64 | 16 | 0.484 | 0.949 | 0.066 | 0.476 | 0.702 | FAIL |

### Unified/factorized and causal-loss ablations

| method/condition | h | direction | magnitude | semantic | output | sign | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| factorized_cache_pca_int_d256 | 1 | 0.179 | 0.331 | 0.074 | 0.236 | 0.620 | FAIL |
| factorized_cache_pca_int_d384 | 1 | 0.207 | 0.344 | 0.072 | 0.210 | 0.440 | FAIL |
| factorized_cache_pca_int_d512 | 1 | 0.289 | 0.384 | 0.084 | 0.314 | 0.600 | FAIL |
| factorized_cache_pca_no_int_d256 | 1 | 0.173 | 0.342 | 0.064 | 0.237 | 0.560 | FAIL |
| factorized_cache_pca_no_int_d384 | 1 | 0.227 | 0.345 | 0.082 | 0.259 | 0.580 | FAIL |
| factorized_cache_pca_no_int_d512 | 1 | 0.346 | 0.409 | 0.102 | 0.293 | 0.580 | FAIL |
| factorized_causal_composite_int_d256 | 1 | 0.745 | 0.744 | 0.386 | 0.679 | 0.680 | FAIL |
| factorized_causal_composite_int_d384 | 1 | 0.762 | 0.752 | 0.396 | 0.692 | 0.680 | FAIL |
| factorized_causal_composite_int_d512 | 1 | 0.816 | 0.818 | 0.432 | 0.770 | 0.740 | FAIL |
| factorized_causal_composite_no_int_d256 | 1 | 0.699 | 0.676 | 0.348 | 0.611 | 0.680 | FAIL |
| factorized_causal_composite_no_int_d384 | 1 | 0.743 | 0.730 | 0.376 | 0.709 | 0.640 | FAIL |
| factorized_causal_composite_no_int_d512 | 1 | 0.808 | 0.807 | 0.464 | 0.749 | 0.760 | FAIL |
| factorized_effect_weighted_multistep_int_d256 | 1 | 0.792 | 0.809 | 0.414 | 0.770 | 0.660 | FAIL |
| factorized_effect_weighted_multistep_int_d384 | 1 | 0.833 | 0.825 | 0.494 | 0.789 | 0.720 | FAIL |
| factorized_effect_weighted_multistep_int_d512 | 1 | 0.849 | 0.855 | 0.492 | 0.821 | 0.740 | FAIL |
| factorized_effect_weighted_multistep_no_int_d256 | 1 | 0.774 | 0.749 | 0.416 | 0.721 | 0.700 | FAIL |
| factorized_effect_weighted_multistep_no_int_d384 | 1 | 0.815 | 0.784 | 0.446 | 0.771 | 0.680 | FAIL |
| factorized_effect_weighted_multistep_no_int_d512 | 1 | 0.840 | 0.832 | 0.508 | 0.805 | 0.720 | FAIL |
| factorized_h1_direction_int_d256 | 1 | 0.803 | 0.815 | 0.430 | 0.772 | 0.800 | FAIL |
| factorized_h1_direction_int_d384 | 1 | 0.835 | 0.857 | 0.454 | 0.804 | 0.700 | FAIL |
| factorized_h1_direction_int_d512 | 1 | 0.852 | 0.875 | 0.522 | 0.821 | 0.740 | FAIL |
| factorized_h1_direction_no_int_d256 | 1 | 0.760 | 0.730 | 0.376 | 0.713 | 0.740 | FAIL |
| factorized_h1_direction_no_int_d384 | 1 | 0.817 | 0.797 | 0.470 | 0.776 | 0.760 | FAIL |
| factorized_h1_direction_no_int_d512 | 1 | 0.845 | 0.853 | 0.518 | 0.802 | 0.800 | FAIL |
| factorized_manifold_regularized_int_d256 | 1 | 0.761 | 0.750 | 0.374 | 0.707 | 0.740 | FAIL |
| factorized_manifold_regularized_int_d384 | 1 | 0.762 | 0.752 | 0.396 | 0.692 | 0.680 | FAIL |
| factorized_manifold_regularized_int_d512 | 1 | 0.816 | 0.818 | 0.432 | 0.770 | 0.740 | FAIL |
| factorized_manifold_regularized_no_int_d256 | 1 | 0.699 | 0.676 | 0.348 | 0.611 | 0.680 | FAIL |
| factorized_manifold_regularized_no_int_d384 | 1 | 0.751 | 0.725 | 0.366 | 0.664 | 0.700 | FAIL |
| factorized_manifold_regularized_no_int_d512 | 1 | 0.808 | 0.807 | 0.464 | 0.749 | 0.760 | FAIL |
| factorized_multistep_direction_int_d256 | 1 | 0.790 | 0.784 | 0.432 | 0.749 | 0.720 | FAIL |
| factorized_multistep_direction_int_d384 | 1 | 0.836 | 0.841 | 0.472 | 0.790 | 0.720 | FAIL |
| factorized_multistep_direction_int_d512 | 1 | 0.859 | 0.880 | 0.516 | 0.827 | 0.760 | FAIL |
| factorized_multistep_direction_no_int_d256 | 1 | 0.760 | 0.732 | 0.418 | 0.729 | 0.720 | FAIL |
| factorized_multistep_direction_no_int_d384 | 1 | 0.808 | 0.798 | 0.478 | 0.767 | 0.800 | FAIL |
| factorized_multistep_direction_no_int_d512 | 1 | 0.849 | 0.847 | 0.508 | 0.810 | 0.780 | FAIL |
| factorized_output_h1_int_d256 | 1 | 0.657 | 0.651 | 0.318 | 0.601 | 0.660 | FAIL |
| factorized_output_h1_int_d384 | 1 | 0.705 | 0.712 | 0.350 | 0.655 | 0.760 | FAIL |
| factorized_output_h1_int_d512 | 1 | 0.727 | 0.705 | 0.364 | 0.665 | 0.740 | FAIL |
| factorized_output_h1_no_int_d256 | 1 | 0.650 | 0.629 | 0.278 | 0.589 | 0.700 | FAIL |
| factorized_output_h1_no_int_d384 | 1 | 0.710 | 0.685 | 0.356 | 0.650 | 0.700 | FAIL |
| factorized_output_h1_no_int_d512 | 1 | 0.718 | 0.691 | 0.376 | 0.651 | 0.720 | FAIL |
| factorized_semantic_h1_int_d256 | 1 | 0.766 | 0.758 | 0.394 | 0.716 | 0.640 | FAIL |
| factorized_semantic_h1_int_d384 | 1 | 0.756 | 0.757 | 0.392 | 0.688 | 0.640 | FAIL |
| factorized_semantic_h1_int_d512 | 1 | 0.812 | 0.816 | 0.456 | 0.766 | 0.840 | FAIL |
| factorized_semantic_h1_no_int_d256 | 1 | 0.700 | 0.678 | 0.334 | 0.623 | 0.680 | FAIL |
| factorized_semantic_h1_no_int_d384 | 1 | 0.752 | 0.729 | 0.364 | 0.704 | 0.720 | FAIL |
| factorized_semantic_h1_no_int_d512 | 1 | 0.811 | 0.800 | 0.448 | 0.748 | 0.800 | FAIL |

Stage-2 result:

GATED: no h1-qualified method.

### Independent h1/h2/h4/h8/h16 causal fidelity

| method/condition | h | direction | magnitude | semantic | output | sign | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| factorized_multistep_direction_int_d512 | 1 | 0.881 | 0.873 | 0.534 | 0.835 | 0.900 | FAIL |
| oracle_factorized_pca_d512 | 1 | 0.292 | 0.383 | 0.110 | 0.263 | 0.740 | FAIL |
| oracle_joint_pca_d512 | 1 | 0.926 | 0.964 | 0.594 | 0.895 | 0.860 | FAIL |
| unified_causal_d512 | 1 | 0.929 | 0.988 | 0.630 | 0.899 | 0.820 | FAIL |
| unified_causal_d512 | 2 | 0.828 | 0.997 | 0.332 | 0.788 | 0.760 | FAIL |
| unified_causal_d512 | 4 | 0.676 | 1.005 | 0.186 | 0.607 | 0.739 | FAIL |
| unified_causal_d512 | 8 | 0.516 | 1.005 | 0.074 | 0.526 | 0.800 | FAIL |
| unified_causal_d512 | 16 | 0.541 | 1.017 | 0.102 | 0.528 | 0.653 | FAIL |

### Architecture ceiling gaps

# Architecture-resolved ceiling — V11

- Ceiling A: combined block-normalized raw dual-PCA, 599D.
- Ceiling B: separate REC + conv + KV coordinates, 1797D.
- Ceiling C: raw full persistent intervention, whose causal identity fidelity is 1 by definition.

| target | combined 599 | architecture 1797 | B − A | metric |
|---|---:|---:|---:|---|
| h1_j_effect | 0.530 | 0.642 | +0.112 | direction_cosine |
| h4_j_effect | 0.190 | 0.230 | +0.040 | direction_cosine |
| output_effect | 0.503 | 0.610 | +0.107 | correlation |

The old 599D result is therefore called a **combined-reference ceiling**, not a complete raw
persistent-state ceiling.

Machine records: `results/v11/processed/architecture_ceiling_v11.parquet` (`ef74609c1c412286a35643f199dc0c4bc03df72fdac54aaa4527f35b4cf7f41a`).


### Adjudication

- Decision-tree outcome: **C**.
- Strongest warranted conclusion: tested low-rank oracle projections do not establish writable sufficiency.
- Smallest causally validated dimension: `None`.
- Hypothesis status: **H2**.
- Autonomous-controller training authorized: **False**.
- Free continuation executed: **False**.
- Manifold record: `results/v11/processed/causal_state_manifold_v11.parquet`.
- Amplification record: `results/v11/processed/causal_error_amplification_v11.parquet`.

### Exact commands

- `bash scripts/run_causal_geometry_v11.sh freeze-base`
- `bash scripts/run_causal_geometry_v11.sh select-models`
- `bash scripts/run_causal_geometry_v11.sh prepare-development`
- `bash scripts/run_causal_geometry_v11.sh freeze-prepared`
- `bash scripts/run_causal_geometry_v11.sh channel-audit`
- `bash scripts/run_causal_geometry_v11.sh oracle-development`
- `bash scripts/run_causal_geometry_v11.sh factorized-stage1`
- `bash scripts/run_causal_geometry_v11.sh freeze-stage2`
- `bash scripts/run_causal_geometry_v11.sh factorized-stage2`
- `bash scripts/run_causal_geometry_v11.sh freeze-confirm`
- `bash scripts/run_causal_geometry_v11.sh prepare-confirmatory`
- `bash scripts/run_causal_geometry_v11.sh confirmatory`
- `bash scripts/run_causal_geometry_v11.sh analyze`
- `bash scripts/run_causal_geometry_v11.sh report`

### V11 changed/generated files

- `artifacts/causal_geometry_v11.freeze.json`
- `artifacts/causal_geometry_v11_confirmatory.freeze.json`
- `artifacts/causal_geometry_v11_manifold_amendment.freeze.json`
- `artifacts/causal_geometry_v11_prepared.freeze.json`
- `artifacts/causal_geometry_v11_stage2.freeze.json`
- `artifacts/v10_immutable.sha256.json`
- `configs/causal_geometry_v11.yaml`
- `reports/ARCH_RESOLVED_CEILING_V11.md`
- `reports/CAUSAL_ERROR_AMPLIFICATION_V11.md`
- `reports/CAUSAL_STATE_MANIFOLD_AUDIT_V11.md`
- `reports/CHANNELWISE_CAUSAL_DECODER_AUDIT_V11.md`
- `reports/FACTORIZED_CAUSAL_STATE_V11.md`
- `reports/ORACLE_LOWRANK_CAUSAL_STATE_V11.md`
- `results/v11/processed/adjudication_v11.json`
- `results/v11/processed/allocation_search_v11.parquet`
- `results/v11/processed/architecture_ceiling_v11.json`
- `results/v11/processed/architecture_ceiling_v11.parquet`
- `results/v11/processed/causal_confirmatory_v11.json`
- `results/v11/processed/causal_confirmatory_v11.parquet`
- `results/v11/processed/causal_error_amplification_v11.json`
- `results/v11/processed/causal_error_amplification_v11.parquet`
- `results/v11/processed/causal_state_manifold_v11.json`
- `results/v11/processed/causal_state_manifold_v11.parquet`
- `results/v11/processed/causal_state_manifold_v11_amendment_1.json`
- `results/v11/processed/causal_state_manifold_v11_amendment_1.parquet`
- `results/v11/processed/channel_compatibility_v11.parquet`
- `results/v11/processed/channel_interaction_nonadditivity_v11.json`
- `results/v11/processed/channel_interaction_nonadditivity_v11.parquet`
- `results/v11/processed/channelwise_causal_v11.json`
- `results/v11/processed/channelwise_causal_v11.parquet`
- `results/v11/processed/confirmatory_states_v11.json`
- `results/v11/processed/development_states_v11.json`
- `results/v11/processed/factorized_causal_stage1_v11.json`
- `results/v11/processed/factorized_causal_stage1_v11.parquet`
- `results/v11/processed/factorized_causal_stage2_v11.json`
- `results/v11/processed/factorized_causal_stage2_v11.parquet`
- `results/v11/processed/method_specs_v11.json`
- `results/v11/processed/oracle_lowrank_causal_v11.json`
- `results/v11/processed/oracle_lowrank_causal_v11.parquet`
- `results/v11/raw/analyze-v11-20260916T130456Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/analyze-v11-20260916T130911Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/causal-v11-20260916T122416Z-0f38a5ef-s20260828/channel_audit_progress.json`
- `results/v11/raw/causal-v11-20260916T122416Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/causal-v11-20260916T123047Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/causal-v11-20260916T123047Z-0f38a5ef-s20260828/oracle_lowrank_causal_projection_progress.json`
- `results/v11/raw/causal-v11-20260916T125009Z-0f38a5ef-s20260828/factorized_causal_stage1_progress.json`
- `results/v11/raw/causal-v11-20260916T125009Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/causal-v11-20260916T125529Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/causal-v11-20260916T125538Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/causal-v11-20260916T125555Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/causal-v11-20260916T125710Z-0f38a5ef-s20260828/independent_causal_confirmatory_progress.json`
- `results/v11/raw/causal-v11-20260916T125710Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/prepare-v11-20260916T121550Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/prepare-v11-20260916T121642Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/prepare-v11-20260916T121849Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/prepare-v11-20260916T121849Z-0f38a5ef-s20260828/prepare_progress.json`
- `results/v11/raw/prepare-v11-20260916T122402Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/prepare-v11-20260916T125610Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/prepare-v11-20260916T125610Z-0f38a5ef-s20260828/prepare_progress.json`
- `results/v11/raw/v11-manifold-amendment-20260916T130850Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/v11-manifold-amendment-20260916T130854Z-0f38a5ef-s20260828/manifest.json`
- `schemas/protocol-v11-record.schema.json`
- `scripts/run_causal_geometry_v11.sh`
- `scripts/run_causal_geometry_v11_manifold_amendment.sh`
- `src/jclosure/experiments/analyze_v11.py`
- `src/jclosure/experiments/analyze_v11_amendment.py`
- `src/jclosure/experiments/causal_v11.py`
- `src/jclosure/experiments/prepare_v11.py`
- `src/jclosure/protocol_v11.py`
- `src/jclosure/protocol_v11_manifold_amendment.py`
- `src/jclosure/reporting_v11.py`
- `src/jclosure/reporting_v11_amendment.py`
- `src/jclosure/state_models_v11.py`
- `tests/test_v11.py`
- `tests/test_v11_manifold_amendment.py`
<!-- V11_END -->


<!-- V11_MANIFOLD_AMENDMENT_1 -->
### V11 manifold metric amendment 1

The clean-zero relative cycle metric in the original V11 manifold record is undefined
because its denominator is zero. It is superseded only for that row by
`causal_state_manifold_v11_amendment_1`; nonzero teacher/decoded metrics are unchanged.
Channel joint-error nonadditivity is additionally reported as a frozen norm proxy.
Amendment freeze: `f17f62532f1136c9cc73109ecf55e2f6689b55638313979eb9c1d5068ac5c14f`. Commands:

- `bash scripts/run_causal_geometry_v11_manifold_amendment.sh freeze`
- `bash scripts/run_causal_geometry_v11_manifold_amendment.sh analyze`
- `bash scripts/run_causal_geometry_v11_manifold_amendment.sh report`

<!-- V12_START -->
## V12 — local causal geometry and intrinsic-dimension audit

Formal decision: **V12-A — MORE_DATA_REQUIRED**. Hypothesis remains **H2**. Smallest independently validated writable dimension: **None**. Autonomous controller authorized: **False**.

### Required scientific answers

1. Semantic metric instability materially explains V11: **False**.
2. 512D data/rank saturation: **not identified**; only train sizes `[600]` had adequate centered rank, so `MORE_DATA_REQUIRED`.
3. Local causal effective rank: stable `1.81`, entropy-effective `3.99`, restricted median r90/r95/r99 `4/6/14` within the frozen 64-direction probe.
4. Causal/PCA overlap: see `VARIANCE_VS_CAUSAL_GEOMETRY_V12.md`; conclusions are restricted to the 64-direction operator.
5. Low-variance/high-causal directions: **True**.
6. Tangent rotation is large under the frozen rule: **True**.
7. Locally-low-dimensional/globally-curved state established: **False** unless both low local rank and successful local oracle hold; current outcome does not authorize that claim.
8. Local causal oracle clearly exceeds global PCA: adjudicated from the independent table, but **no method is authorized** unless listed here: `[]`.
9. A 128/256/384/512D local candidate passes h1: **False** (512D local itself was rank-limited).
10. h2/h4/h8 pass: **False** under the all-family frozen gates.
11. Dominant trajectory failure: **direction_rotation_and_semantic_divergence**; finite-horizon ratios were not treated as eigenvalues.
12. Strict full-state replacement succeeded: **False**.
13. Writable dimension is proven higher than predictive dimension: **not proven globally**; current restricted causal spectrum and failed writeback remain consistent with a larger writable state.
14. Strongest supported outcome: **V12-A — MORE_DATA_REQUIRED**.
15. H2 remains: **True**.
16. Upgrade to H3: **False**.
17. Autonomous controller authorization: **False**.

The learned state-dependent decoder was not run because the protocol did not authorize it before causal geometry and strict replacement succeeded.
<!-- V12_END -->

<!-- V13_START -->
## V13 — expanded causal bank, probe scaling, and path geometry

Formal decision: **V13-F — HIGH-DIMENSIONAL WRITABLE STATE**.

- New bank sizes: `{'final_test': 250, 'train': 4800, 'validation': 500}`; independent final was not used for selection.
- Restricted r95 curve m=64/128/256/512: `{'64': 5.0, '128': 6.5, '256': 8.0, '512': 9.0}`.
- Stable low local causal rank under the frozen expanded probes: `True`.
- Low-variance/high-causal directions: `False`.
- Same-prompt rank-16 tangent angle: `29.18°`.
- Moving tangent improves static: `True`; independent h1/h2/h4/h8 pass: `False`.
- Practical local linear radius: `None`.
- Instantaneous/cumulative median r95: `8.0` / `11.0`.
- Smallest independently validated writable dimension: `None`.
- Complete replacement state: **False** (`ABSOLUTE_REPLACEMENT_NOT_YET_TESTABLE_FROM_CURRENT_DELTA_REPRESENTATION`).
- H2 remains: **True**. H3 authorized: **False**.
- Autonomous controller authorized: **False**.

Historical V1–V12 conclusions remain frozen. V13 causal-edit results are not described as absolute state replacement, and restricted exact-JVP rank is not described as full raw-state intrinsic dimension.
<!-- V13_END -->

<!-- V14_START -->
## V14 — Finite Causal Control and Tangent Transport

Formal procedural outcome: **V14-STOP — NUMERICAL_INTERFACE_GATE_FAILED**. The BF16 persistent-state writeback creates a measured finite-effect floor; the frozen all-target exact-JVP/finite-difference equivalence gate did not pass through ε=2. `MIN_CAUSAL_EFFECT_NORM = 0.00680280`. SNR relabeling explains part of V13's small-alpha anomaly, not the later-horizon failures. First- and second-order valid radii: `None` / `None`. Train-atlas transport and holonomy are measurable, but they do not establish a global compact coordinate system.

Development closed-loop causal steering was run on ten validation cases; independent confirmation was **not** run because no numerically validated finalist was available. V14-A through V14-E are not fully established. H2 remains; H3 is not supported. Absolute state replacement and autonomous controller training are **not authorized**. V1–V13 conclusions remain frozen. See `reports/V14_COMPLETE_REPORT.md` for the full report bundle.
<!-- V14_END -->

<!-- V15_START -->
## V15 — Quantization-Aware Causal Actuation

Formal status: **V15-STOP — FINITE_RESPONSE_LINEARITY_GATE_FAILED**. Real BF16 writeback dead-zones were quantified per REC/Conv/K/V; only 16/100 adaptive state×direction×channel combinations met the predeclared finite-effect selection rule. At accurate realized state perturbations, some output targets still disagree with frozen exact JVP, supporting tested-regime V15-B. The restricted finite-response matrix has higher r95 than ideal JVP (V15-D pattern), but the frozen finite-response linearity gate failed, so spectra and actuator-aware closed-loop results remain diagnostic/development rather than validated finite control. No eligible finalist or new independent confirmatory bank was created. H2 remains; H3 and absolute state replacement are not supported; autonomous-controller training is not authorized. The required cumulative-report append invalidates one old V14 whole-file integrity test; V15's guard confirms all other historical bytes are unchanged. See `reports/V15_COMPLETE_REPORT.md` for all standalone reports, machine records, frozen gates and limitations.
<!-- V15_END -->

<!-- V16_START -->
## V16 — Nonlinear Finite Causal Action Geometry

Formal procedural status: **V16-STOP — NONLINEAR_RESPONSE_VALIDATION_GATE_NOT_PASSED**. A frozen five-family 100-train/50-validation finite-action bank was measured with real BF16 writeback; reliable fraction 0.891. Odd/even, scale, pair, requested-versus-realized, nonlinear model hierarchy, observed spectra, finite action order, primitive coverage and empirical h1 J reachability are reported in the single-file `reports/V16_COMPLETE_REPORT.md`. No k∈{2,4,8,16,32} model passed the required J and normalized-stack strict heldouts, so no validated compact nonlinear causal actuator dimension was identified. This does not establish V16-F/nonexistence: realized-coordinate, primitive-sequence and dynamic reachable-set coverage are limited. Nonlinear MPC and new independent confirmation were gated off. H2 remains; H3 candidate/full H3, absolute replacement and autonomous controller are not authorized. V1–V15 records remain frozen.
<!-- V16_END -->

<!-- V17_START -->
## V17 — Interventional State Sufficiency

Formal procedural outcome: **V17-STOP — STATE_CONTEXT_CEILING_GATE_NOT_PASSED**. Clean REC/Conv/KV raw-state linear-kernel context on the reused V16 reliable single-action bank improved held-out J/normalized-stack relative L2 by only **0.0126/0.0113** over J+action, below the frozen 0.05 materiality gate; a raw-only reference also did not reach it. REC yielded the largest individual ceiling gain, but fixed-base cross-fitted raw-residual corrections did not produce stable improvement. This does **not** establish J sufficiency or absence of useful persistent context under other models. The compact-context sweep, unseen-action sufficiency, h2/h4/h8, transitions and independent final bank were gated off. Smallest sufficient C dimension: **none identified**. H2 remains; H3 candidate state, absolute replacement and autonomous state-model training are **not authorized**. V1–V16 frozen results are unchanged; see `reports/V17_COMPLETE_REPORT.md` for standalone reports, records and limitations.
<!-- V17_END -->

<!-- V18_START -->
## V18 — Strong State-Context Ceiling and Horizon Localization

Formal outcome: **V18-STOP — STRICT_MATCH_NOT_IDENTIFIED_AND_NO_MATERIAL_RAW_CEILING**. The new crossed bank contains 2000 train and 400 validation states under eight shared signed finite action coordinates; h1/h2/h4/h8 use a fixed 400/80-state panel. The unified nonlinear h1 full-raw versus J-only absolute J/stack relative-L2 gains are **0.0008/-0.0011** (frozen material gate 0.05; passed: False).
Full-raw stack gain by horizon h1/h2/h4/h8: **-0.0014, -0.0055, -0.0188, 0.0061**. Earliest material horizon: **None**. Strict matching: **MATCH_NOT_IDENTIFIED**; history material: **False**. Compact search: **COMPACT_CONTEXT_SEARCH_NOT_AUTHORIZED**. Independent final: **UNOPENED_PENDING_FROZEN_FINALIST**.
H2 remains. H3 candidate state and autonomous state-model training are not authorized without a response-sufficient compact state and independent V18-F confirmation. These are finite-action response results, not proof of a complete state or physical replacement. V1–V17 frozen records are unchanged. See `reports/V18_COMPLETE_REPORT.md` for all standalone reports and machine-record integrity index.
<!-- V18_END -->

<!-- V19_START -->
## V19 — Counterfactual Workspace Sufficiency and Natural Dynamics

Formal development outcome: **V19-B — CURRENT_J_IS_NOT_SUFFICIENT_FOR_TESTED_FINITE_ACTION_RESPONSE_CONTEXT (development)**. Active boundary-held-J q states and matched finite probe actions were crossed on 400 train/100 validation base states (five families); primary h1 M/R 0.7899, state-bootstrap CI [0.7577, 0.8097], natural N/R0 1.3284. Matched actuator rate 1.0000. h1 raw-context gain natural/action -0.0018/0.0133. C_response/C_dynamics search authorized: True/False; restricted C_response search ran with selected k=128. Independent final: UNOPENED_NO_ELIGIBLE_FINALIST_AFTER_RESTRICTED_COMPACT_SEARCH. H2 remains; H3 and autonomous state-model training are not authorized. No complete state or absolute replacement claim. See `reports/V19_COMPLETE_REPORT.md` for all standalone reports and integrity index.
<!-- V19_END -->

<!-- V20_START -->
## V20 — Compact Causal Response Operator State

Independent V19-B confirmation passed on **250** disjoint base states: same J, distinct P, matched finite action, M/R **0.7862** (95% state-bootstrap CI **[0.7782, 0.7931]**). At smaller reliable αq=0.25, all five selected active q retained nonzero descriptive modulation.

The crossed V20 response bank contains **150 training** and **50 validation** base states, each with natural P0 plus three same-J Pq states, crossed with **18** shared measured directions and both signs; **6** final directions remain sealed. The train-action oracle fingerprint is `Φ_train(P)=[R_P(a1),…,R_P(a12)]`, with 288-D normalized response per action. Five models × k=2–128 were tested; best validation diagnostic model `bilinear_latent_operator` at k=128 had unseen-direction stack relative L2 **0.9273**, unseen-sign **1.0056**. The full frozen direction/sign/scale/composition gate did **not** pass; `k_operator_min` is **not identified**.

Finite operator pair geometry: median principal angle **37.4441°**, gain Pq/P0 **1.0399**, r95 change **0.0000**. Operational same-J workspace-state aliasing observed: **True**. Historical V13 JVP rank is not a paired V20 subspace test.

Formal result: **V20-D_NO_COMPACT_OPERATOR_DIMENSION_IDENTIFIED**. Raw P→C encoder, conditional raw-gain equivalence, channel encoder audit and independent V20 final were **not eligible / unopened**, not passed or failed empirical tests. `V21_DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`; H2 remains; H3 and autonomous training remain unauthorized. No physical cache replacement is licensed.

Protocol digest `8726bddd18d48665856ed35caefe4a32b25216ffddd578732b7f34565644069c`; action hashes `{"final_heldout": "cbfa6ce7c46aa4c400260c5a6fede1ea47999046e51d940f30db20e92b310aa2", "train": "af8cc82763286b93496d38a2401283e4fa9a5cb0e2c8524afb879d3f03bbb6c2", "validation": "9495edef649915c5bd58930f58f8d3960929661c1abcff20580d7af7686b8192"}`; V20 processed integrity index `results/v20/processed/v20_integrity_index.json`. See `reports/V20_COMPLETE_REPORT.md` for every standalone report in one file.

Append-only adjudication clarification: a separate frozen nonlinear k=128 model reached seen-direction/new-state relative L2 **0.2337** but failed unseen directions at **1.3318**. Thus **V20-C_ACTION_SPECIFIC_OPERATOR_ENCODING_ONLY** is a secondary descriptive finding, while the primary formal outcome remains V20-D. This correction is recorded in `results/v20/processed/v20_adjudication_amendment_1.json` with freeze digest `ed58a8a8314daa9bd6b2358b0a7565f48305a4a3440b66a8268a8d30d512dffc`; no data, model, threshold or gate was changed.

Test audit: 228 passed, 2 legacy cumulative-report hash tests failed (V14/V16); all six V20 tests passed. These old expectations already differed from the V19 parent commit, and the frozen old manifests were not modified. See `results/v20/processed/v20_test_audit.json`.
<!-- V20_END -->

<!-- V21_START -->
## V21 — Action Coordinate Ceiling and Paired Causal Operator Geometry

Formal outcome: **V21-F_CROSS_ACTION_OPERATOR_REMAINS_UNIDENTIFIED_UNDER_TESTED_PRACTICAL_FAMILIES + V21-G_ACTION_DATA_LIMITED + V21-E_SHARED_STATE_DEPENDENT_CAUSAL_GEOMETRY**. Holding S2/G2 fixed, V20 Z0→full requested Z1 reduces unseen-direction relative L2 from **1.3137** to **0.6386**, but unseen sign is **0.9371** and no practical candidate passes the frozen cross-action gate. S1 k128→full S2 gains only **0.0023** L2 under fixed Z1/G2. The Z1 validation train-span residual is **0.7437**, and 4→12 training actions improve L2 **0.3578**. Thus current action coverage remains a material limit; this does not establish compact-state nonexistence.

On 50 development + 25 disjoint validation P0/Pq bases, matched 64-probe exact-JVP and central finite operators have validation median r95 **5.0/7.0**, P0/Pq median rotations **36.7976°/30.4461°**, rotation Spearman **0.5669**, and P0 JVP–finite subspace overlap **0.8550**. Strong shared-geometry gate: **True**. Z4/Z5 are same-action diagnostic oracles, not deployable coordinates.

`COMPACT_OPERATOR_SEARCH_REOPENED=False`; `RAW_TO_OPERATOR_ENCODER_AUTHORIZED=False`; `DYNAMIC_STATE_SEARCH_AUTHORIZED=False`. Final six action responses remain sealed. H2 remains; H3, complete replacement and autonomous control are not authorized. Historical V1–V20 frozen inputs are unchanged. See `reports/V21_COMPLETE_REPORT.md` for every standalone report, amendment, machine-record hash, and limitation.
<!-- V21_END -->


## V22 — Causal Action Manifold Expansion and Input-Side Operator Geometry

Formal outcome: `V22-D_ACTION_DATA_LIMITED`. Metric-corrected input subspaces rotate with persistent state, but no practical action coordinate passes; action scaling remains unsaturated, compact-operator reopening is false, both final sets remain sealed, H2 remains, and H3/dynamic-state search remain unauthorized. See `reports/V22_COMPLETE_REPORT.md`.

---

# V23 Complete Report

## Identity

- Name: **Oracle Local Action Charts and Input-Rank Scaling**
- Parent: `c755b4d7b0baf3aa0291456fcf49169087b8c863`
- Protocol hash: `378297e7c0413b1bcdddc05894dbcaff23835cdb349567dceb73ebb8bd387c66`
- State split hash: `762f62fee5c470ab719983cbad34b5822a2dde33dc130a33101cb1b9f48c035b`
- Probe hashes: `{"128": "e1daff6b0e3ac35c47e3c4d4780a55211f4250aca98b14a530fd6ab08083c450", "256": "24d3d90c2dc3fe7cac33f625789b1d855d89eeb5a5fbf1a76861288a83dbce03", "512": "9a1dd253da8e52b3163942f1a22bd2a65573e34522f7654782f7a34b05f48804", "64": "1dbe804f7b4c3e0c7cd12449c7ce760c3381a0d7a9689c05a9077275052e722b"}`
- Train / held-out / new-final hashes: `9a1dd253da8e52b3163942f1a22bd2a65573e34522f7654782f7a34b05f48804` / `793cab02928568a7201c863993943c76da566809bb51fb6ce7f573e91fa14d5f` / `df6f69a7b5b07f8af9ec1e460d6bb63f3ffe61cc79a3c92812d4ed552a1e2b2b`

## Outcome

Formal outcome: **V23-E_INPUT_RANK_NOT_SATURATED+V23-F_ORACLE_CHART_INSUFFICIENT**.

The tested low-rank subspace chart is insufficient for held-out finite-action prediction; this does not imply that local charts do not exist. Input-sensitive causal rank remains probe-limited if saturation fails.

## Rank scaling

| m | Gram rank | condition | JVP r90/r95/r99 | finite r90/r95/r99 |
|---:|---:|---:|---:|---:|
| 64 | 64 | 9.046e+04 | [11.0, 17.0, 31.5] | [15.5, 21.0, 33.0] |
| 128 | 128 | 7.690e+05 | [15.0, 23.0, 48.0] | [18.5, 27.0, 46.5] |
| 256 | 254 | 2.979e+06 | [17.5, 28.0, 65.0] | [23.0, 33.5, 60.5] |
| 512 | 417 | 5.413e+06 | [19.0, 31.0, 75.0] | [25.0, 36.5, 67.5] |

Joint rank saturation: **FALSE**.

## Oracle charts

Best JVP: `k24:M4_cubic_nonlinear`; best finite: `k24:M0_nearest`; `k_chart_min=N/A`.

### Best finite metrics

| category | relative L2 | cosine | J cosine | norm ratio |
|---|---:|---:|---:|---:|
| unseen_amplitude | 0.960954 | 0.317000 | 0.310984 | 0.201072 |
| unseen_dense | 0.769178 | 0.597264 | 0.570696 | 0.715100 |
| unseen_direction | 1.025821 | 0.377751 | 0.338953 | 0.727023 |
| unseen_pair | 0.118783 | 0.806665 | 0.880796 | 0.996332 |
| unseen_sign | 1.019546 | 0.393395 | 0.344085 | 0.724646 |

### Best JVP metrics

| category | relative L2 | cosine | J cosine | norm ratio |
|---|---:|---:|---:|---:|
| unseen_amplitude | 0.878402 | 0.646534 | 0.626799 | 0.207469 |
| unseen_dense | 0.452086 | 0.866666 | 0.843087 | 0.836195 |
| unseen_direction | 0.843700 | 0.662272 | 0.683380 | 0.680571 |
| unseen_pair | 0.113035 | 0.811872 | 0.856977 | 0.876703 |
| unseen_sign | 0.827548 | 0.653629 | 0.672629 | 0.690202 |

JVP/finite input overlap: `0.359687`; angle: `55.258445°`.

## Smoothness, aliasing, transport

- Nearest-J clean-state chart angle: `51.313420°`; adjacent-token result unavailable on the frozen panel.
- Same-J P0/Pq chart angle: `51.726420°`.
- Procrustes transport fidelity: `0.337425`.

## Prediction and adaptation

- Chart-predictor status: `NOT_RUN_ORACLE_GATE_FAILED`; S0/S1/S2/S4 = `N/A/N/A/N/A/N/A`.
- Local probe curve: `{"128": {"best_token": "k32:M0_nearest", "median_cosine": 0.38607414960140807, "median_j_cosine": 0.348008725962357, "median_norm_ratio": 0.7142199043858382, "relative_l2": 1.019415099502293, "sample_count": 3200}, "16": {"best_token": "k12:M0_nearest", "median_cosine": 0.35119735147429276, "median_j_cosine": 0.3351858789811696, "median_norm_ratio": 0.7528161745833588, "relative_l2": 1.013987146551028, "sample_count": 3200}, "32": {"best_token": "k32:M0_nearest", "median_cosine": 0.363111439163039, "median_j_cosine": 0.3862278914622397, "median_norm_ratio": 0.8187013260238863, "relative_l2": 1.0223800648995869, "sample_count": 3200}, "4": {"best_token": "k4:M2_ridge", "median_cosine": 0.4979512772478455, "median_j_cosine": 0.41367253496208634, "median_norm_ratio": 0.6790563808784552, "relative_l2": 0.9264408380315695, "sample_count": 3200}, "64": {"best_token": "k64:M0_nearest", "median_cosine": 0.37766823683066963, "median_j_cosine": 0.39511790983697914, "median_norm_ratio": 0.8210619182045444, "relative_l2": 1.0152519252826655, "sample_count": 3200}, "8": {"best_token": "k8:M2_ridge", "median_cosine": 0.5194881231497745, "median_j_cosine": 0.4695532470665056, "median_norm_ratio": 0.6793416466948523, "relative_l2": 0.9312359388926131, "sample_count": 3200}}`.
- Oracle vs learned direction L2: `0.843700` vs `1.001623`.

## Authorization

- `COMPACT_OPERATOR_SEARCH_REOPENED = FALSE`
- `RAW_TO_OPERATOR_ENCODER_AUTHORIZED = FALSE`
- `H2_REMAINS = TRUE`
- `H3_AUTHORIZED = FALSE`
- `DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`
- Independent final: `FROZEN_AND_UNOPENED_NO_DEVELOPMENT_VALIDATION_FINALIST`

## Verification

V23 tests: `6 passed`. Full suite: `244 passed, 2 inherited failures`. Historical records were not rewritten to hide inherited failures.

## Files

All required standalone V23 reports are in `reports/`; machine-readable JSON/Parquet outputs are in `results/v23/processed/`; raw response matrices remain in the external V23 scratch directory.

---

# V24 Complete Report

## Identity

- Name: **Causal Output Bottleneck Validation and Layer Localization**
- Parent: `4dcc412d85fb2995237c48ec755beb34fd662941`
- Protocol hash: `36ecce8f74a00071b4cae5293ba13a9db7aabc53274cd7d6d8531ab42af3c904`
- State hashes: `{"development": "ffdd34027b9881b637b85f1cca2dab58891770711a3c80cc6ce3e03893949950", "independent_final": "545c291dcae7b69bb6265cacc31b1c153fe36e03f0932568bfb881b75864726f", "mediation_basis": "cdde0503b5d03fef89ea9482292fd9fb960e6153370a6df965e6ec6c0db79279", "mediation_validation": "3608bd70eb2b3b2d51fc08a3fd883318dc1961c55332e593ca9620c5ac7727cc", "natural_transition": "65ee72641565a91f68ad6a9f3a41fbb4469bb7bfd8d9f6a13df9e9e136a79fec", "validation": "03546c5f088c31e094e56dc9683706dadc800cdc8ac9b7361917efa1e196bcbb"}`
- Action hashes: `{"heldout": "793cab02928568a7201c863993943c76da566809bb51fb6ce7f573e91fa14d5f", "mediation": "789d44b22902b45a7d811b9f2df8354ff279b20039f6a7f194a07539e8f3a5fc", "train": "764b2fa64b26b1ad9fb1831c6bb32315b44e0f08674f729ce5e0c7e765dc7467"}`
- Target projection hash: `99e08cd12c9b3751e251c36c6352a16fc6c98951021cbb1abca7b7b2b50d26f9`
- Candidate layer/k: `31/24`
- Mediation protocol hash: `029b32788b0e22e14d7a62a813532d25f360e7c8aeabf0a05e0ab7780484cfa5`
- Mediation-results amendment hash: `45564a3ba9071f66b61b7ca6f4a918d32c989cb8612472607ac20f25bd49ba31`
- Natural transitions: `2` eligible h2−h1 rows; `8` audited h1-only exclusions from ten frozen design rows.

## Outcome

Formal outcome: **V24-F_DISTRIBUTED_CAUSAL_MEDIATION**.

Representational bottleneck pass: `FALSE`. Causal mediation gate: `FALSE`. A causal mediator is not equated with a complete model state.

## Compression profile

- V23 input JVP/finite r95@256: `28.0/33.5`.
- Early/candidate/late broad hidden r95: `1.0/1.0/1.0`.
- Late T0/vocabulary r95: `1.0/1.0`.
- Formal collapse layer: `N/A`.

## Held-out coverage

| basis | explained norm | residual | cosine |
|---|---:|---:|---:|
| B_FAMILY | 0.847354 | 0.390700 | 0.920724 |
| B_GLOBAL | 0.832632 | 0.409102 | 0.912632 |
| B_LOCAL_J | 0.604833 | 0.628618 | 0.777834 |
| B_LOCAL_P | 0.727686 | 0.521835 | 0.853064 |
| B_STATE_ORACLE | 0.999457 | 0.023293 | 0.999949 |
| CONTROL_RANDOM | 0.099770 | 0.948804 | 0.315820 |
| CONTROL_RANDOM_CAUSAL | 0.737741 | 0.512111 | 0.859573 |
| CONTROL_SHUFFLED_FAMILY | 0.478126 | 0.722403 | 0.691895 |
| CONTROL_VARIANCE | 0.235224 | 0.874506 | 0.485021 |

## Causal branches

- B_ONLY L2/cosine/norm: `0.501006/0.936193/1.298022`.
- PERP_ONLY retained, 97.5% upper: `1.375170/1.389140`.
- FULL-minus-B retained: `0.797728`.
- Restoration mediated fraction: `-1.382951`.
- Transplant L2/cosine/norm: `2.290694/0.209903/2.348547`.
- Regeneration ratio: `N/A`.
- Horizon status: `H1_FAILED_NO_H2_H4_H8_OPENING`.

## Authorization

- `H2_REMAINS = TRUE`
- `H3_AUTHORIZED = FALSE`
- `DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`
- Independent final: `FROZEN_AND_UNOPENED_NO_SINGLE_DEVELOPMENT_VALIDATION_MEDIATOR`

## Verification

V24 tests: `6 passed in 28.72s`. Full suite: `2 failed, 250 passed, 3 warnings in 42.17s (inherited V14/V16 FINAL_REPORT hash assertions)`. Historical cumulative-report hash failures remain visible.

---

# V25 Complete Report

## Identity and frozen scope

- Name: **Distributed Causal Interaction and Response Convergence**
- Parent commit: `2197d58dc6699d3a6bba467f703f267fa3dcabc7`
- Base protocol hash: `1889c4d8b1534302a241b2082fec2b243483b454ee9bc01b9749acfacfd8b8ad`
- Design freeze: `a4f3c479997194f52ef6c974a2d8e365049952394796a79cb1fa1c47ca442d28`
- Component-basis freeze: `204b6e88d6b90d2aa75af953f656d0861d3d33202e2072548bb1aa8dd0fb3144`
- Strict interaction amendment: `0236f6e4735091c1cb35c799b30d1e3ae218c76c0924167f1a2605aa039d0f72`
- Zero-rank amendment: `8be5b300f21c4c6aedc9fd814f349198e796a8d14ef236015af96e97bf5f47a2`
- Adjudication freeze: `cf08a168a4695704ed5c2678c12875d732f2c9350b06413edc96203be686c8eb`
- Independent final: 50 states, 10 per family, hash `bf2998344c5c3d2e4da4579b1db3a85f7886e2eabc12b262708129af34bb9fcf`, frozen and unopened.

## Formal result

**V25-B_CAUSAL_CANCELLATION_SUPPORTED+V25-E_READOUT_COMPRESSION_ONLY**.

`V25-A=FALSE`, `V25-B=TRUE`, `V25-C=FALSE`, `V25-D=FALSE`, `V25-E=TRUE`, `V25-F=FALSE`, `V25-G=FALSE`, `V25-H=FALSE`.

The supported picture is causal cancellation plus low-dimensional behavioral/readout compression without a demonstrated low-dimensional causal realization or internal convergence.

## Rank audit

Raw uncentered broad responses reproduce r95=1 at every layer. State-wise intervention centering yields r95 13–14, and centered plus row-normalized responses yield r95 24–25. Thus V24's rank-one result is measurement-definition-specific and does not establish a general one-dimensional causal geometry.

## Strict distributed interaction

The corrected estimand is `Y00=clean; Y10=clean+B; Y01=clean+C; Y11=clean+(B+C)` over `160` rows. Median interaction ratio is `0.340985` with bootstrap lower bound `0.319602`. Interaction centered r95 is `77`. However, the strict writeback gate fails (minimum B cosine `0.633638`), so V25-A is false. Cancellation index `0.257974` supports V25-B.

## Layerwise response geometry

Exact-zero matrices at layers 0–23 have rank zero. The first nonzero layer is 24. Pooled centered r95 rises from `56` at layer 24 to `99` at layer 31, and every family rises. Many-to-one response convergence and internal causal convergence are therefore false. Frozen mapping layers before 31 are zero-response and cannot identify a meaningful contraction map.

## Distributed realization and readout

No dimension among `[4, 8, 16, 32, 64, 128, 256]` meets the same-J realization criteria; at dimension 256, relative L2 is `0.568966`. In contrast, centered r95 is `47` for direct full hidden, `19` for J-128, `19` for logits-32, and `15` for semantic-32. This supports output/readout compression only, not compact internal state.

## Other diagnostics

- Background-conditioned response coordinates: `FALSE`.
- Channel factorial medians: `{"ConvxKV": 0.730175992084515, "RECxConv": 1.317024583107477, "RECxConvxKV": 1.2131817544658459, "RECxKV": 1.0587004306136611}`.
- Order relative-difference median/max: `0.435503/0.934490`.
- Multihorizon status: `NOT_OPENED_NO_FROZEN_ALL_FAMILY_TWO_TOKEN_CONFIRMATORY_PANEL`.
- Interaction/convergence correlations are association-only and are not causal proof.

## Append-only corrections

1. Effective-rank values were clamped to ambient dimension without overwriting original records.
2. The formal interaction estimand was corrected to clean+B+C for Y11; the persistent-FULL records remain historical diagnostics.
3. Exact-zero response matrices were assigned rank zero; the earlier cumulative-energy artifact was retained but superseded.

## Authorization and verification

- `H2_REMAINS = TRUE`
- `H3_AUTHORIZED = FALSE`
- `DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`
- Historical final opened: `FALSE`
- V25 independent final opened: `FALSE`
- V25 tests: `6 passed in 5.89s`
- Full suite: `256 passed, 2 failed, 3 warnings in 77.35s (inherited V14/V16 FINAL_REPORT hash checks)`; the two failures are inherited V14/V16 cumulative-report hash checks.

---

# V26 Complete Report

## Identity

- Name: **Temporal Readout Potency of Persistent State**
- Parent: `04f70cebcb66a9bc2431052b44c488b1ce608d07`
- Protocol hash: `6f3471d17dd87cf5dcd61d123c07fc8ec172e207dad1a59412c1f21c45125870`
- Design hash: `44569092ecfe8e31d15687b89f2397637be249070034a92933d936d1cefffb5b`
- Current-distal bank hash: `237a978f9d048e07f824655b9813ea254ac7651b05b23d04c8c957cd00edae8c`
- Final adjudication hash: `92dc7851dd2731575b17cdde956281eb92d61c47b5487c15bf22f508c30b0771`
- Diagnostics hash: `ed1190db5e185efcb072907139e7e9f2bc66185b6ebb77ccf36be55aae19dc3c`

## Prospective design

Development, validation, and independent final contain 100, 50, and 50 disjoint balanced states. The h0-only bank was frozen before any h1–h8 response was observed. It contains 1,170 requested perturbations, of which 1,152 are reliable and strict current-silent. Repeated-forward p99 noise is zero for all six targets.

## Formal result

**V26-A_TEMPORAL_READOUT_POTENCY_CONFIRMED+V26-B_STRICT_CURRENT_SILENT_FUTURE_POTENCY+V26-C_SAME_WORKSPACE_FUTURE_DIVERGENCE_CONFIRMED+V26-D_ROTATING_FUTURE_POTENT_GEOMETRY**.

`V26-A=TRUE`, `V26-B=TRUE`, `V26-C=TRUE`, `V26-D=TRUE`, `V26-E=FALSE`, `V26-F=FALSE`, `V26-G=FALSE`, `V26-H=FALSE`, `V26-I=FALSE`.

The supported interpretation is that the tested persistent-state interventions create future-relevant causal distinctions that are absent from the current readout but become visible at the next and later autoregressive steps. This is not a memory-variable, compact-state, or complete-dynamical-state claim.

## Temporal potency

| development | 1 | 1.012791 | 1.000000 | 1.000000 | 5/5 |
| development | 2 | 1.082126 | 1.000000 | 1.000000 | 5/5 |
| development | 4 | 1.060365 | 1.000000 | 1.000000 | 5/5 |
| development | 8 | 1.059371 | 1.000000 | 1.000000 | 5/5 |
| validation | 1 | 1.009252 | 1.000000 | 1.000000 | 5/5 |
| validation | 2 | 1.029356 | 1.000000 | 1.000000 | 5/5 |
| validation | 4 | 1.062481 | 1.000000 | 1.000000 | 5/5 |
| validation | 8 | 1.034589 | 1.000000 | 1.000000 | 5/5 |

All 1,050 primary development/validation rows emerge at h1. The unique h2 final finalist confirms with median Q `1.104824`, potent fraction `1.000000`, and 5/5 family replication.

## Same-workspace divergence

At h0 all accepted interventions have identical J and all other current outputs. Median normalized future J effect is approximately 0.92–1.03 across roles and horizons, and essentially all rows cross the re-entry threshold. Therefore current workspace/readout causal sufficiency is rejected for the tested intervention class and controlled continuation.

## Future geometry

Future r95 is h1 `61`, h2 `57`, h4 `47`, and h8 `52`. Adjacent projector distances are `1->2:0.429579, 2->4:0.505247, 4->8:0.523711`. Geometry rotates; U0 has rank zero and does not explain future responses.

## Secondary findings and limitations

- Channel interaction ratios are substantial but diagnostic; formal V26-F fails family replication.
- State-dependence CV is below the frozen threshold; V26-G is false.
- Smooth scaling and odd sign symmetry are not supported.
- Exact temporal cache-state JVP and natural-transition alignment were not established and are not used in gates.
- Shuffled state-specific transplant and readout-only controls were not technically comparable under this shared-coordinate, post-readout intervention boundary; numerical-scale, same-norm random, ordinary finite, and clean-replay controls are reported.

## Authorization

- `H2_REMAINS = TRUE`
- `H3_AUTHORIZED = FALSE`
- `DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`
- `CAUSAL_ROUTING_V27_AUTHORIZED = TRUE`
- Autonomous controller work remains unauthorized.

## Verification

V26 tests: `7 passed in 2.19s`. Full suite: `263 passed, 2 failed, 3 warnings in 58.10s (inherited V14/V16 FINAL_REPORT hash checks)`; both failures are inherited cumulative-report hash checks.

---

# V27 Complete Report

## Identity

- Name: **Pre-Readout Counterfactuals and Next-Token Causal Re-Entry**
- Parent: `0657f731e8a165425baa231d4880bab36bf01169`
- Protocol hash: `1de32375db20b199876b0ce5077056393b95cf392c216b757eb6600d1973a59e`
- Design hash: `c7ac19b558d54cdf6fc561c70f7ae1e5207d6c85e02a1dce676c4ea2617176b7`
- Initial bank hash: `09fa769987745722f535d0382096eca99b3d8903f794eced2f191ce78b35a130`
- Expanded bank hash: `d657c1c23622dfbffc87a7a0db9c6beec11739c06e1b7389c16483f44b87939d`
- Adjudication hash: `4e885a2cb4fd83e0a6c6e3f27a8ed53e378fbd388a7a5ffdef8c940928db6342`

## Prospective design and boundary

The intervention was moved to the latest valid pre-readout boundary: the cache after all prompt tokens except the final token. The model then executed the final prompt token normally, with no J/logit/semantic/residual restoration or clamping. All six repeated-forward numerical floors were exactly zero. An ordinary intervention materially changed current output, confirming invariance was not built into the boundary.

The four disjoint balanced panels contain 25 calibration, 50 development, 25 validation, and 25 unopened independent-final states. All boundaries, targets, silence thresholds, candidates, routing layers/components, gates, and opening rules were frozen before response observation.

## Primary result

**V27-F_V26_BOUNDARY_SPECIFIC_ONLY**.

`V27-A=FALSE`, `V27-B=FALSE`, `V27-C=FALSE`, `V27-D=FALSE`, `V27-E=FALSE`, `V27-F=TRUE`, `V27-G=FALSE`, `V27-H=FALSE`.

The initial architecture-resolved bank tested 975 rows; the frozen V19 same-J expansion tested 3,600 rows. Of 4,575 total rows, 4,460 passed actuation reliability. None met `PRE_READOUT_SILENT` and none met `PRE_READOUT_DISTAL`. The minimum reliable aggregate h0 Q by family ranged from `0.347766` to `0.463103`, well above the 0.10 distal ceiling.

Therefore V27 found no evidence that a tested persistent distinction can be introduced before current readout, survive the full natural current-token computation, remain current-invisible, and then reappear at h1. V26 remains valid as a post-readout future-state result, but its stronger hidden-current-state interpretation is unsupported by this panel.

## Timing diagnostic

For a predeclared joint action on 10 states, median pre-readout h0 Q was `1.153113` versus post-readout `0`; h1 Q was `0.929026` versus `1.063260`. The h1 direction cosine was `0.608038`. This diagnostic confirms both boundaries can affect the future but only the post-readout boundary guarantees current silence.

## Sequential stop

`DETAILED_ROUTING_AUTHORIZED = FALSE`. Consequently no layer trace, first causal site, restoration, transplant, factorial, workspace reconstruction, re-entry geometry, natural comparison, or independent final was opened. Empty machine-readable routing artifacts make this stop explicit and prevent absence from being mistaken for a null routing result.

## Authorization

- `H2_REMAINS = TRUE`
- `H3_AUTHORIZED = FALSE`
- `DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`
- `AUTONOMOUS_CONTROLLER_AUTHORIZED = FALSE`

## Verification

V27 tests: `6 passed in 1.76s`. Full suite: `268 passed, 3 failed, 3 warnings in 52.61s`; the three failures are inherited cumulative-report hash checks.

---

# V28 Complete Report

## Identity and frozen starting point

**Token-Level Persistent State Read/Write Transaction**. Parent `83235995faa5aefb0f9fd9dab6f865a3506df2e1`; protocol `b4a0b8cc34e3caf98bacb885157c24f23db88cf54c497c8cc5f0da1629d4363f`; boundary `d38905cfdd0e9480d59751a084f578f59f1dc546823c42adf91c32a4b6298d87`; adjudication `583e9895d58f65ee4eb62f93883826260ad407ae72f97ced723dc3e152a13f2d`. V27-F remains the starting point: V26's current silence was post-readout, whereas tested pre-readout perturbations changed current output. V28 does not revive pre-existing current-silent memory, compact state, or H3.

## Architecture and prospective design

In Qwen3.5, six layers have native REC/Conv state and two have attention KV. After the complete natural token-t forward, layer-30 current readout and outgoing cache are both available. Exact incoming REC/Conv fields can replace the natural outgoing fields without touching current activations; KV cannot be shortened to the old sequence length without changing positional semantics, so its new-slot copy is a separately labeled diagnostic. All future branches use identical next tokens.

Disjoint balanced panels were frozen at 25 calibration, 75 development, 50 validation, and 50 independent-final states. One development pilot state was explicitly removed from formal analysis after the layer-23 versus layer-30 J naming ambiguity was discovered; primary results use 74 development states and layer-30 readout-J at both t and t+1. The pilot future observation and unchanged thresholds are recorded in an append-only endpoint amendment.

## Formal outcomes

`V28-A=TRUE`, `V28-B=TRUE`, `V28-C=TRUE`, `V28-D=FALSE`, `V28-E=FALSE`, `V28-F=FALSE`, `V28-G=FALSE`, `V28-H=FALSE`.

The predeclared **exact REC+Conv natural-write block** left the already-computed current readout unchanged (Q=0) while altering next-token readout by median Q `34.551789` development, `36.818478` validation, and `35.872796` independent final. All five families replicated, all formal writebacks were exact, and the potent fraction was 1.0 in each panel. Incoming-state read interventions, by contrast, changed current output. Natural-write dose effects declined monotonically as the retained REC/Conv update approached 100%.

Natural same-prompt donor branches showed exact **full outgoing-cache** transfer under both reciprocal directions, with cosine ≈1 and magnitude ratio 1. This is a complete-cache identity control: making the future cache equal to a donor and replaying the same token should reproduce that donor. It establishes full-state transplant fidelity/transferability but does **not** identify a selective single-channel route. Single REC, Conv, KV, and REC+Conv partial transplants did not pass the reciprocal donor-direction gate. Individual write blocks nevertheless had large marginal effects, so neither a purely distributed-only necessity claim nor a global channel ranking is justified.

## Workspace and horizons

Next-token layer-30 J changes under write block by median Q `89.562832` development and `91.570364` validation despite fixed current J. This supports that next workspace depends causally on the prior outgoing REC/Conv commit, not that it is fully constructed by or identical to a compact state. h1/h2/h4 median Q is `32.265/7.898/3.448` development and `36.671/8.854/3.671` validation.

## Limits and interpretation

The zero current write effect is guaranteed by the after-readout interception boundary; the nontrivial causal result is that blocking **the natural outgoing update**, rather than adding an arbitrary future perturbation, changes subsequent computation and survives dose/replication/final tests. Old-state restoration can be off-manifold; same-family shuffled full cache also causes large future changes. Exact full-cache transfer is expected, and partial transfer failure prevents a localized-carrier claim. Predictive write features, norm-matched random write, and a matched small-write causal control were not established and are excluded from formal gates. No compact or complete-state inference is made.

`H2_REMAINS=TRUE`; `H3_AUTHORIZED=FALSE`; `DYNAMIC_STATE_SEARCH_AUTHORIZED=FALSE`; `AUTONOMOUS_CONTROLLER_AUTHORIZED=FALSE`.

V28 tests: `6 passed in 1.73s`. Full suite: `274 passed, 3 failed, 3 warnings in 51.90s`; three inherited cumulative-report hash checks fail.

---

# V29 Complete Report

## Identity and frozen history

**Causal Content of Natural Persistent-State Writes — What Does a Token Commit to the Future?** Parent `8d3954ef05957ab22f27261ea7eaa217497b2b62`; base protocol `86389aa651802da41d09c0217c9f588586eab45a7274aee4d5d1ae42c2114913`; adjudication `f404213f48100effb31d7aa99dfc5141faacb7dfdb3eca83f47c67f0802b858a`. V28's exact REC+Conv write block and full-cache identity control remain historical. V29 does not reinterpret V28 failed partial transfers as successful carriers.

## Prospective same-incoming fork

The frozen 25/75/50/50 balanced panels exclude V28 IDs. At each prompt prefix, top-plausible distinct-surface token A/B were selected from the model distribution **before** their writes or future responses were observed. Both branches start with the identical native cache and complete their current-token computation naturally. The next input token is the same frozen V18 teacher token. Current readout and outgoing state are captured before any transplant. Same-token calibration replay was exact.

An explicit interface amendment preserves a failed preliminary calibration: reusing V28's posterior 6 REC/Conv + 2 KV layers as a purported full V29 cache yielded median relative L2 `0.823`. A transient 53-state development run was interrupted before formal partition output. No threshold, token pair, state role, or finalist changed. Formal V29 covers **all 24 recurrent/conv layers and 8 attention layers**; full cache transplant is bitwise donor-equal and response-exact.

## Causal results

Natural token forks generated distinct writes and shared-next-token future Q median `90.529` development, `92.302` validation, `92.745` independent final. All exact native-field writebacks passed. Full-cache transfer is an identity ceiling, not carrier localization.

| channel | role | A←B cosine / magnitude / rel-L2 | B←A cosine / magnitude / rel-L2 | reciprocal gate |
|---|---|---:|---:|---:|
| REC | development | 0.062 / 0.095 / 0.999 | 0.259 / 0.132 / 0.980 | False |
| REC | validation | -0.002 / 0.095 / 1.003 | 0.358 / 0.137 / 0.962 | False |
| Conv | development | 0.988 / 0.971 / 0.163 | 0.992 / 0.989 / 0.127 | True |
| Conv | validation | 0.987 / 0.962 / 0.175 | 0.992 / 0.993 / 0.125 | True |
| KV | development | 0.082 / 0.067 / 0.997 | 0.108 / 0.067 / 0.995 | False |
| KV | validation | 0.088 / 0.067 / 0.996 | 0.093 / 0.070 / 0.996 | False |
| REC+Conv | development | 0.998 / 0.995 / 0.067 | 0.998 / 0.997 / 0.067 | True |
| REC+Conv | validation | 0.998 / 0.996 / 0.070 | 0.998 / 0.996 / 0.067 | True |

Conv-only and REC+Conv donor writes pass reciprocal development and validation gates; REC-only and KV-only fail. The **predeclared REC+Conv** finalist passed the 50-state independent final (`A←B` median cosine `0.998`, `B←A` `0.997`). KV remained recipient-native in this transfer. Thus the tested natural write commits future-relevant, token-conditioned recurrent/conv content beyond exact KV token-history storage. Conv alone passes, so a strict distributed-only/no-single-channel claim is false. These experiments do not decode a semantic variable inside a channel.

Ten states per role tested four pre-write next-token probes: REC+Conv median donor cosine was `0.988` development and `0.986` validation (A←B), with full cache exact. Native REC+Conv greatly outperformed norm-related random, shuffled and sign-flipped controls. h1/h2/h4 donor-specific effects were measured only after h1 gates. Cross-state factorials found aligned but imperfectly transferable write contrasts; neither global nor state-dependent code is formally settled.

## Workspace, dimension, and limits

Held-out diagnostic R² for token+J future prediction was `0.423`; adding coarse write geometry gave `0.488`. This predictive gain is **not** a J-matched causal proof of workspace incompleteness. Current J is a readout, not a persistent cache field.

Train-only REC+Conv write-effect PCA tested k `[4, 8, 16, 32, 64]` on held-out development and validation; qualified k: `32, 64`. k `[128, 256]` exceeded the train span and were not tested. Any qualifying k would be a write-effect approximation, **not** a complete or dynamical state dimension. No arbitrary perturbation is promoted to a natural write.

Formal outcomes:
- `V29_A_TOKEN_CONDITIONED_STATE_WRITE_CONFIRMED`: **True**
- `V29_B_SAME_BACKGROUND_WRITE_TRANSFER_CONFIRMED`: **True**
- `V29_C_RECURRENT_WRITE_CARRIES_NONTRIVIAL_CONTENT`: **True**
- `V29_D_KV_DOMINATED_TOKEN_CARRYOVER`: **False**
- `V29_E_DISTRIBUTED_WRITE_CONTENT`: **False**
- `V29_F_STATE_DEPENDENT_WRITE_CODE`: **False**
- `V29_G_GLOBAL_WRITE_CODE`: **False**
- `V29_H_CURRENT_WORKSPACE_INCOMPLETE_FOR_WRITE`: **False**
- `V29_I_COMPACT_WRITE_EFFECT_DIMENSION_IDENTIFIED`: **True**
- `V29_J_NO_COMPACT_WRITE_EFFECT_DIMENSION_IDENTIFIED`: **False**
- `V29_K_WRITE_CONTENT_REMAINS_UNIDENTIFIED`: **False**

`H2_REMAINS=TRUE`; `H3_AUTHORIZED=FALSE`; `DYNAMIC_STATE_SEARCH_AUTHORIZED=FALSE`; `AUTONOMOUS_CONTROLLER_AUTHORIZED=FALSE`. V29 tests: `6 passed in 1.78s`. Full suite: `280 passed, 3 failed, 3 warnings in 52.90s`. Full machine-readable Parquet/JSON/NPZ records, state/token/write/transplant/future hashes and the integrity index accompany this report.
