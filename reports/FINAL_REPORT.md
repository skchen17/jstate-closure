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
