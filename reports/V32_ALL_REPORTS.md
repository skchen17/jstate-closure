# V32 — All Reports in One File

The 21 standalone V32 reports are reproduced below in frozen order.

---

<!-- 01: V32_REC_CORRECTION_REPLICATION.md; sha256=9daecc4a7cb947c3057b47414f4bfb872ce4303b99ef927d8bfbf468a7cbce9c -->

# REC Correction Replication — V32

Exact native factorials used new 100 development and 50 validation states, one primary ordinary natural token fork per state, six prewrite probes, and recipient-native KV. The independent 50-state final remains sealed. Relative improvement is `(||Ydonor−Y01||−||Ydonor−Y11||)/||Ydonor−Y01||`; state-bootstrap lower bounds use 2,000 resamples.

| role | n | positive fraction | median reduction | bootstrap LB | median alignment | alignment LB | families | replication gate | alignment gate |
|---|---|---|---|---|---|---|---|---|---|
| development | 100 | 1.000 | 0.410 | 0.346 | 0.809 | 0.759 | 5 | True | True |
| validation | 50 | 1.000 | 0.372 | 0.309 | 0.779 | 0.723 | 5 | True | True |

| role | family | n | positive fraction | median reduction | median alignment |
|---|---|---|---|---|---|
| development | boolean_logic | 20 | 1.000 | 0.372 | 0.780 |
| development | modular_arithmetic | 20 | 1.000 | 0.520 | 0.880 |
| development | short_graph_traversal | 20 | 1.000 | 0.365 | 0.777 |
| development | simple_state_transition | 20 | 1.000 | 0.431 | 0.824 |
| development | variable_binding | 20 | 1.000 | 0.324 | 0.739 |
| validation | boolean_logic | 10 | 1.000 | 0.355 | 0.759 |
| validation | modular_arithmetic | 10 | 1.000 | 0.433 | 0.824 |
| validation | short_graph_traversal | 10 | 1.000 | 0.341 | 0.752 |
| validation | simple_state_transition | 10 | 1.000 | 0.409 | 0.807 |
| validation | variable_binding | 10 | 1.000 | 0.362 | 0.770 |

| role | frozen token category | n | median reduction | median alignment | positive fraction |
|---|---|---|---|---|---|
| development | CJK_SURFACE_UNRESOLVED | 18 | 0.677 | 0.947 | 1.000 |
| development | FUNCTION_WORD | 19 | 0.196 | 0.599 | 1.000 |
| development | LEXICAL_SURFACE | 23 | 0.480 | 0.854 | 1.000 |
| development | NUMERIC | 25 | 0.384 | 0.803 | 1.000 |
| development | PUNCTUATION_OR_STRUCTURE | 15 | 0.365 | 0.774 | 1.000 |
| validation | CJK_SURFACE_UNRESOLVED | 11 | 0.650 | 0.937 | 1.000 |
| validation | FUNCTION_WORD | 11 | 0.260 | 0.673 | 1.000 |
| validation | LEXICAL_SURFACE | 10 | 0.356 | 0.764 | 1.000 |
| validation | NUMERIC | 12 | 0.320 | 0.738 | 1.000 |
| validation | PUNCTUATION_OR_STRUCTURE | 6 | 0.344 | 0.754 | 1.000 |

REC-only remains weak: relative donor L2 medians development/validation 0.998/0.997; Conv-only is 0.283/0.288, joint is 0.163/0.180. This establishes a conditional causal *effect of exact REC field replacement* on the tested donor-fidelity endpoint, not a localized mechanism or REC as an independent carrier. Sources: `factorial_*_v32.parquet`, matching vectors and audits.

---

<!-- 02: V32_GAIN_ROTATION_CORRECTION_DECOMPOSITION.md; sha256=bfcb80b005c9204631f9235e16318d4b4fb4eccf87ea5de02429dcb7a3b4cca7 -->

# Gain, Rotation and Correction Decomposition — V32

Development/validation joint gain ratios (norm RC / norm C) are 1.012/1.013. Per-row scalar alpha projects the joint move onto Conv; this is an oracle descriptive upper bound, not a prospective predictor. On validation the median scalar-gain donor L2 is 0.287, versus true joint 0.180; true joint beats that scalar in 1.000 of rows.

The fixed global rotation comparison fits a rank-32 orthogonal Procrustes map on development moves only, retaining identity outside that development subspace. Validation median donor L2 is 0.280, versus true joint 0.180; joint wins in 1.000 of rows. This rejects only the tested low-capacity fixed rotation, not every possible context-dependent rotation.

The sequential decomposition of conditional REC response on development gives median normalized gain component 0.114, residual-target component 0.805, and remaining orthogonal component 0.583. These overlapping norm fractions do not sum to one; they are descriptive, whereas exact factorial error reduction is causal.

---

<!-- 03: V32_CONTEXTUAL_REC_CORRECTION.md; sha256=3f12b0a21cc1d5ade1479709376b0d195301385c588e11aaf22dea0e62240a17 -->

# Contextual REC Correction — V32

For each of 10 development and 10 validation targets, target Conv and recipient KV are fixed. Matched REC is compared with recipient, wrong-token, same-family wrong-state, cross-family wrong-state, shuffled, sign-flipped and random same-norm REC. All variants use the same six probes.

| role | REC condition | median relative donor L2 |
|---|---|---|
| development | CROSS_FAMILY_WRONG_STATE_REC | 0.547 |
| development | MATCHED_REC | 0.161 |
| development | RANDOM_SAME_NORM_REC | 1.292 |
| development | RECIPIENT_REC | 0.321 |
| development | SAME_FAMILY_WRONG_STATE_REC | 0.351 |
| development | SHUFFLED_REC | 2.419 |
| development | SIGN_FLIPPED_REC | 0.522 |
| development | WRONG_TOKEN_REC | 0.318 |
| validation | CROSS_FAMILY_WRONG_STATE_REC | 0.522 |
| validation | MATCHED_REC | 0.202 |
| validation | RANDOM_SAME_NORM_REC | 1.488 |
| validation | RECIPIENT_REC | 0.287 |
| validation | SAME_FAMILY_WRONG_STATE_REC | 0.367 |
| validation | SHUFFLED_REC | 2.329 |
| validation | SIGN_FLIPPED_REC | 0.446 |
| validation | WRONG_TOKEN_REC | 0.306 |

Matched REC beats all four primary off-context controls in development 1.000 and validation 1.000 of targets. Cross-state REC is off-manifold; its failure alone does not identify state-specific gating. The mapping was frozen before context-specific responses but after factorial observations, so V32-C is **not** promoted to a fully preregistered formal outcome. Sources: `context_mapping_*_v32.json`, `context_*_v32.parquet`.

---

<!-- 04: V32_CROSS_STATE_REC_CORRECTION.md; sha256=fcfad125af8ea02f52932f3b3facc8262eb7907f34708af7d8b2a7ee1af34de8 -->

# Cross-State REC Correction — V32

Same-family and cross-family wrong-state source IDs are in frozen `context_mapping_*_v32.json`. Development matched/same-family/cross-family median relative donor errors are 0.161/0.351/0.547; validation values are 0.202/0.367/0.522. This is target-probe cross-state correction fidelity. Direct effect-space cosine between states is **not comparable** because the six frozen probe IDs differ by state; it is not reported as though measured. Different incoming states can place REC off the natural target manifold. Therefore this is a bounded transfer-failure observation, not proof that REC stores a specific missing content or that a gate is state-specific.

---

<!-- 05: V32_CROSS_TOKEN_REC_CORRECTION.md; sha256=86a8ceb6cfa6ed21ff654e7c9b1e8c0dbfe0be275d028f4e2316739356a898fa -->

# Cross-Token REC Correction — V32

The design froze two candidate contrasts per state, from different response-blind surface categories where eligible. Secondary factorial contrasts were evaluated in 25 development and 10 validation states, without fitting. The primary library also repeats token IDs across independent incoming states.

| role | distinct primary tokens | tokens in ≥2 states | max states/token | secondary contrasts | secondary median reduction | median within-state absolute difference |
|---|---|---|---|---|---|---|
| development | 38 | 29 | 6 | 25 | 0.418 | 0.208 |
| validation | 31 | 15 | 3 | 10 | 0.461 | 0.305 |

Within-state primary/secondary comparisons share probes and are directly comparable as donor-error reductions. Cross-state response-vector cosines are not, because each state's probe IDs differ. Category-stratified results appear in `V32_REC_CORRECTION_REPLICATION.md`. The observed variation is not a fitted token/state interaction law or REC content code. The full per-row records and token/state hashes are in `factorial_*_v32.parquet` and `design_v32.json`.

---

<!-- 06: V32_NEXT_TOKEN_CORRECTION_TRACE.md; sha256=9af0b6078547062a220d72b48fe59ebabf22fe8c62f879f3620613ac9d910afe -->

# Next-Token Correction Trace — V32

On the frozen 10+10 tracing states, one of six prewrite probes was recorded across all 32 post-block residual outputs and 24 recurrent layers' qkv projection, raw gate-a/b projection, normalized recurrent read and recurrent output. The primary causal endpoints remain six-probe signatures.

| role | architecture boundary | rows | median conditional norm | median boundary residual alignment |
|---|---|---|---|---|
| development | gate_a | 240 | 1.084 | 0.948 |
| development | gate_b | 240 | 0.707 | 0.955 |
| development | normalized_recurrent_read | 240 | 1.470 | 0.968 |
| development | post_block_residual | 320 | 1.852 | 0.927 |
| development | qkv_projection | 240 | 12.041 | 0.921 |
| development | recurrent_output | 240 | 0.802 | 0.968 |
| validation | gate_a | 240 | 1.080 | 0.969 |
| validation | gate_b | 240 | 0.678 | 0.960 |
| validation | normalized_recurrent_read | 240 | 1.353 | 0.974 |
| validation | post_block_residual | 320 | 1.682 | 0.949 |
| validation | qkv_projection | 240 | 12.012 | 0.949 |
| validation | recurrent_output | 240 | 0.758 | 0.975 |

Layerwise post-block growth is reported below; norms and cosines are descriptive and are not donor-fidelity interventions.

| role | layer | median conditional norm | boundary target cosine | final residual target cosine |
|---|---|---|---|---|
| development | 0 | 0.110 | 1.000 | -0.013 |
| development | 1 | 0.167 | 1.000 | -0.000 |
| development | 2 | 0.406 | 1.000 | 0.006 |
| development | 3 | 0.516 | 0.907 | -0.002 |
| development | 4 | 0.660 | 0.910 | 0.004 |
| development | 5 | 0.736 | 0.916 | 0.000 |
| development | 6 | 0.907 | 0.952 | -0.007 |
| development | 7 | 0.949 | 0.920 | 0.008 |
| development | 8 | 0.948 | 0.934 | -0.005 |
| development | 9 | 1.056 | 0.954 | -0.002 |
| development | 10 | 1.126 | 0.957 | -0.010 |
| development | 11 | 1.247 | 0.946 | -0.001 |
| development | 12 | 1.274 | 0.944 | 0.002 |
| development | 13 | 1.502 | 0.956 | -0.006 |
| development | 14 | 1.533 | 0.961 | 0.021 |
| development | 15 | 1.679 | 0.941 | 0.029 |
| development | 16 | 1.743 | 0.946 | 0.024 |
| development | 17 | 1.936 | 0.960 | 0.052 |
| development | 18 | 2.211 | 0.965 | 0.073 |
| development | 19 | 2.618 | 0.887 | 0.097 |
| development | 20 | 2.929 | 0.908 | 0.123 |
| development | 21 | 3.221 | 0.909 | 0.144 |
| development | 22 | 3.434 | 0.908 | 0.175 |
| development | 23 | 3.692 | 0.833 | 0.221 |
| development | 24 | 3.780 | 0.848 | 0.238 |
| development | 25 | 3.952 | 0.858 | 0.268 |
| development | 26 | 4.202 | 0.864 | 0.309 |
| development | 27 | 4.538 | 0.781 | 0.340 |
| development | 28 | 5.003 | 0.816 | 0.419 |
| development | 29 | 5.593 | 0.821 | 0.467 |
| development | 30 | 6.773 | 0.844 | 0.594 |
| development | 31 | 9.272 | 0.802 | 0.802 |
| validation | 0 | 0.117 | 1.000 | -0.004 |
| validation | 1 | 0.185 | 1.000 | -0.006 |
| validation | 2 | 0.364 | 1.000 | -0.010 |
| validation | 3 | 0.461 | 0.912 | -0.006 |
| validation | 4 | 0.600 | 0.926 | 0.007 |
| validation | 5 | 0.703 | 0.942 | 0.005 |
| validation | 6 | 0.985 | 0.976 | 0.014 |
| validation | 7 | 1.020 | 0.955 | 0.013 |
| validation | 8 | 1.165 | 0.962 | 0.005 |
| validation | 9 | 1.229 | 0.974 | 0.006 |
| validation | 10 | 1.368 | 0.979 | 0.011 |
| validation | 11 | 1.378 | 0.963 | 0.006 |
| validation | 12 | 1.391 | 0.966 | -0.008 |
| validation | 13 | 1.511 | 0.972 | 0.016 |
| validation | 14 | 1.542 | 0.975 | 0.026 |
| validation | 15 | 1.602 | 0.952 | 0.032 |
| validation | 16 | 1.840 | 0.965 | 0.026 |
| validation | 17 | 1.975 | 0.971 | 0.058 |
| validation | 18 | 2.216 | 0.975 | 0.085 |
| validation | 19 | 2.476 | 0.904 | 0.130 |
| validation | 20 | 2.693 | 0.914 | 0.161 |
| validation | 21 | 2.977 | 0.918 | 0.203 |
| validation | 22 | 3.155 | 0.922 | 0.218 |
| validation | 23 | 3.488 | 0.866 | 0.240 |
| validation | 24 | 3.817 | 0.877 | 0.264 |
| validation | 25 | 3.941 | 0.883 | 0.282 |
| validation | 26 | 4.175 | 0.887 | 0.321 |
| validation | 27 | 4.678 | 0.821 | 0.353 |
| validation | 28 | 5.179 | 0.829 | 0.427 |
| validation | 29 | 5.739 | 0.836 | 0.494 |
| validation | 30 | 6.570 | 0.851 | 0.582 |
| validation | 31 | 8.016 | 0.807 | 0.807 |

Raw gate projections are not themselves gate activation values; sigmoid/softplus inside the kernel was not intercepted. Likewise the functional convolution output and delta-update term were not directly hookable with the existing module interface. Thus trace coverage is partial, and no first *causal* site follows from activation divergence. Machine source: `next_token_trace_*_v32.parquet`.

---

<!-- 07: V32_FIRST_CORRECTION_SITE.md; sha256=f9c4a0bb16526652291a393f474ed71a6e84a50702fc3a9f4745d83f69cd44f0 -->

# First Measurable Correction Site — V32

A recurrent output is called measurable when its Y11−Y01 norm exceeds 1e−6 on the first frozen probe; this is a numerical descriptive threshold, not a noise-calibrated causal threshold.

| role | states | median first layer | min | max |
|---|---|---|---|---|
| development | 10 | 0.000 | 0 | 0 |
| validation | 10 | 0.000 | 0 | 0 |

The development-only candidate rule selected recurrent-output layer 30 by maximal median alignment (0.226) with the final residual donor-minus-Conv target. Validation did not select a new site. The first measurable layer is not the first causal mediator, and the selected site must pass bidirectional intervention to qualify.

---

<!-- 08: V32_CORRECTION_INTERCEPTION.md; sha256=7bd2a3fd3ecfea467f22dbc3f6e1e3d1be547d7ece61416892f3c04addc89868 -->

# Correction Interception — V32

At the frozen layer-30 recurrent-operator output, replace Y11's component with its same-probe Y01 value and continue the suffix. The same input cache, token and position are retained; requested/realized output hashes are equal in all probes. Removed benefit is measured against the six-probe donor error.

| role | n | bidirectional successes | median removed | median restored | remove cosine | restore cosine | families | formal gate |
|---|---|---|---|---|---|---|---|---|
| development | 10 | 0 | 0.249 | 0.161 | 0.314 | 0.328 | 0 | False |
| validation | 10 | 0 | 0.188 | 0.083 | 0.284 | 0.297 | 0 | False |

No formal localized necessity claim is supported. A single candidate was selected prospectively; its failure does not establish that no other local site exists. Source: `site_intervention_*_v32.parquet`, `site_intervention_audit_*_v32.parquet`.

---

<!-- 09: V32_CORRECTION_REVERSE_TRANSPLANT.md; sha256=dca131b823218a2eace0805647e32fee9f3821e309d51c25bded27a42ab69acf -->

# Correction Reverse Transplant — V32

The reverse experiment inserts the matched Y11 recurrent-operator output into the Y01 Conv-only branch, one probe at a time. Exact field hash equality and baseline replay are required. Validation median restored benefit is 0.083, below the frozen ≥0.50 gate; median correction-direction cosine is 0.297, below ≥0.80. Development values are 0.161/0.328. Therefore layer 30 is not a bidirectionally validated correction site; final remains closed.

---

<!-- 10: V32_REC_CONV_LAYER_INTERACTION_MAP.md; sha256=343ca9a92ef8aa0aa201442ff7e216dc8f3c6f9470ab8f0592c71e0fc9ede2f2 -->

# REC × Conv Layer Interaction Map — V32

The frozen four recurrent-layer groups and four Conv-layer groups form 16 exact-native pairings, each tested on 10 development and 10 validation states with six probes. The score is `(||donor−Conv_j||−||donor−REC_i+Conv_j||)/||donor−recipient||`. It is not an anatomical mediation score.

| role | REC group | Conv group | median donor-L2 improvement |
|---|---|---|---|
| development | early | early | 0.011 |
| development | early | early_mid | -0.002 |
| development | early | late | -0.005 |
| development | early | late_mid | -0.006 |
| development | early_mid | early | -0.004 |
| development | early_mid | early_mid | -0.009 |
| development | early_mid | late | -0.008 |
| development | early_mid | late_mid | -0.011 |
| development | late | early | 0.012 |
| development | late | early_mid | 0.009 |
| development | late | late | 0.013 |
| development | late | late_mid | 0.008 |
| development | late_mid | early | 0.001 |
| development | late_mid | early_mid | 0.001 |
| development | late_mid | late | -0.006 |
| development | late_mid | late_mid | -0.001 |
| validation | early | early | 0.005 |
| validation | early | early_mid | -0.002 |
| validation | early | late | -0.005 |
| validation | early | late_mid | 0.001 |
| validation | early_mid | early | -0.010 |
| validation | early_mid | early_mid | -0.003 |
| validation | early_mid | late | -0.010 |
| validation | early_mid | late_mid | -0.001 |
| validation | late | early | 0.008 |
| validation | late | early_mid | 0.007 |
| validation | late | late | 0.009 |
| validation | late | late_mid | 0.007 |
| validation | late_mid | early | 0.005 |
| validation | late_mid | early_mid | 0.005 |
| validation | late_mid | late | 0.002 |
| validation | late_mid | late_mid | -0.006 |

No individual group pair was selected from validation, and no single-layer refinement or formal pairwise gate was frozen. V32-H is therefore not established. Source: `rec_conv_layer_map_*_v32.parquet`.

---

<!-- 11: V32_SAME_VS_CROSS_LAYER_ROUTING.md; sha256=603f66d37fefbfbbd7be843520c35d488438f9edcd19cafbafc8ada037f033b3 -->

# Same- versus Cross-Layer Routing — V32

| role | same group | REC earlier | REC later | all cross group |
|---|---|---|---|---|
| development | 0.005 | -0.005 | 0.004 | -0.000 |
| validation | 0.001 | -0.003 | 0.003 | 0.000 |

These summaries average different partial Conv baselines and are descriptive. Same group is not the same physical layer; group index order does not prove causal anatomy. No predeclared superiority or sufficiency threshold for a group pairing was met/assessed, and no claim that cross-layer routing is required is made.

---

<!-- 12: V32_GATE_MEDIATION.md; sha256=de951cc1cf00f736df94a576feb2393654274b344a54ee0de80fac888dc3c064 -->

# Gate Mediation — V32

Raw gate-a and gate-b projection traces vary under the factorial branches, but sigmoid/softplus gate values and recurrent-kernel update terms were not independently recorded. Diagnostic exact-output patches at the development-nominated layer are below. Component probes were specified after the primary factorial plan and cannot become formal V32-E evidence.

| role | component | median removed | median restored | remove cosine | restore cosine |
|---|---|---|---|---|---|
| development | gate_a | -0.001 | 0.001 | -0.011 | 0.033 |
| development | gate_b | -0.001 | -0.001 | -0.031 | 0.010 |
| validation | gate_a | -0.001 | 0.003 | -0.006 | 0.037 |
| validation | gate_b | 0.001 | -0.002 | 0.014 | 0.024 |

Patching a raw projection output can perturb downstream gate computation; it does not isolate a unique state-specific gate law. V32-E remains unconfirmed.

---

<!-- 13: V32_QKV_UPDATE_MEDIATION.md; sha256=60f70b89f4f4822045798baf35468c6786eeee1f5386dcb7ff876e3b2861586d -->

# QKV and Update Mediation — V32

The architecture exposes a joint pre-convolution qkv projection and a normalized recurrent read. The q/k/v post-convolution split and fused delta-update were not individually writable with the audited interface. Exact diagnostic patches at the frozen primary layer yield:

| role | component | median removed | median restored | remove cosine | restore cosine |
|---|---|---|---|---|---|
| development | normalized_recurrent_read | 0.249 | 0.161 | 0.314 | 0.328 |
| development | qkv_projection | 0.026 | -0.010 | -0.019 | -0.014 |
| validation | normalized_recurrent_read | 0.188 | 0.083 | 0.284 | 0.297 |
| validation | qkv_projection | 0.020 | -0.024 | -0.021 | -0.033 |

These checks do not isolate q, k, v, beta or the update term, and the component list was not part of the original primary candidate-selection rule. V32-F is unconfirmed, not falsified for untested update variables.

---

<!-- 14: V32_RESIDUAL_INTEGRATION.md; sha256=2f2a63a0b8588c48d4a353867adbb07fc8c2674ab275b810315a1266ae586850 -->

# Residual Integration — V32

The diagnostic `post_residual_mlp_normalized_input` patch replaces the post-attention layernorm *output* entering the MLP; it does not replace the entire residual sum.

| role | component | median removed | median restored | remove cosine | restore cosine |
|---|---|---|---|---|---|
| development | post_residual_mlp_normalized_input | 0.156 | 0.126 | 0.398 | 0.407 |
| validation | post_residual_mlp_normalized_input | 0.166 | 0.104 | 0.376 | 0.364 |

Because the residual skip path and recurrent cache update remain separate, this cannot adjudicate whether correction is instantiated at residual addition. V32-G remains unconfirmed.

---

<!-- 15: V32_CORRECTION_DIMENSION.md; sha256=00ba05aeb4feeb6d9bb36863815a0229fe2321eba314a33ef0cbf294994fd9cf -->

# Correction Dimensionality — V32

The centered 100-row development matrix of six-probe conditional REC moves has descriptive r90/r95/r99 = 61/75/92 and effective rank 59.388; its maximal empirical rank is 99. These are ranks of a sampled *response*, not compact state dimensions or transferable content. No rank concentration threshold was frozen for a causal basis and no basis reconstruction intervention was authorized. The optional causal subspace question is unanswered.

---

<!-- 16: V32_MULTI_PROBE_CORRECTION.md; sha256=0020a1d69d0c7c24533133c0bec822a487b2380609601c027fcb5342c8dd5ca9 -->

# Multi-Probe Correction — V32

Every primary factorial, context comparison, 4×4 layer-map row and causal interception/reverse transplant used the concatenated six-probe normalized response containing J, selected logits, semantic log-probabilities, workspace, broad vocabulary and late residual. One-probe sublayer traces were only used for candidate nomination, never as the decisive causal endpoint. Frozen probe IDs and hashes are per state in `design_v32.json`; `factorial_*_v32.parquet` stores the hash on each row. The V32-A/B results therefore survive this frozen multi-probe signature; no generalization to arbitrary future-token distributions is claimed.

---

<!-- 17: V32_MULTI_HORIZON_CORRECTION.md; sha256=067890220c385de146ae22968cbf0288262999b6ef9174c0c3d7a65fb7fc1881 -->

# Multi-Horizon Correction — V32

Primary h1 factorial and localization were completed. The frozen rule permits h2/h4 only after an h1 localized mechanism passes development and validation. The nominated site failed, so h2/h4 were not opened and remain unanswered. No h8 experiment was required. This is protocol gating, not evidence of absent longer-horizon effects.

---

<!-- 18: V32_STRICT_INTERFACE_AUDIT.md; sha256=f5374e4af6e142e99483c4937363a604b92956e668d8a4a6ce5daed23fb0eacb -->

# Strict Interface Audit — V32

| role | factorial rows | native exact | site probe patches | site exact | diagnostic probe patches | diagnostic exact |
|---|---|---|---|---|---|---|
| development | 125 | True | 60 | True | 300 | True |
| validation | 60 | True | 60 | True | 300 | True |

Incoming prefix and cache-channel hashes, donor/recipient full-state hashes, token-pair and six-probe hashes, target Conv and recipient KV hashes, layer-group hashes, requested/realized component hashes, cache lengths and role IDs are in the design and raw audit records. Hooks are context-managed and removed even on failure. V32 final-state response count is zero. The factorial audit's non-outcome `field_count` metadata incorrectly records 120 for the three conditions; an append-only `audit_metadata_amendment_v32.json` corrects this to 24+24+48=96 without altering the raw records, exact equality checks or outcomes. The trace lacks a direct convolution-output and recurrent-kernel update interception; those are explicitly untested, not silently described as measured.

---

<!-- 19: V32_EXECUTION_MANIFEST.md; sha256=adecbb373174ce2f395014a8c9f48159e19a2e5c6da2a784fc01b5b133f570d4 -->

# Execution Manifest — V32

Parent `774a867eb4ffb91ef749e971fb78922ed312a285`; base freeze `48a0ea5b83e8f4d1b62eca245e6766aa0d8b85d5c50c369d99c551e72175f004`. Response-blind panels: 25 calibration, 100 development, 50 validation, 50 unopened independent final. Excluded V28–V31 formal states; 40 ordinary candidate tokens in five surface classes; six frozen probes per state. Primary factorial 100/50 rows, secondary token contrasts 25/10, next-token trace 10/10, context controls 10/10, 4×4 group map 10/10 and bidirectional nominated-site intervention 10/10.

| freeze artifact | digest prefix |
|---|---|
| rec_conv_mechanism_v32.freeze.json | 48a0ea5b83e8f4d1 |
| rec_conv_mechanism_v32_audit_amendment.freeze.json | ead6a5538179183f |
| rec_conv_mechanism_v32_context_development.freeze.json | ff734b7200d34ad2 |
| rec_conv_mechanism_v32_context_mapping_development.freeze.json | edc40938a5ec62b2 |
| rec_conv_mechanism_v32_context_mapping_validation.freeze.json | 267c8f3b14a64318 |
| rec_conv_mechanism_v32_context_validation.freeze.json | f8b382c5fbfd2752 |
| rec_conv_mechanism_v32_design.freeze.json | c8536822d1461dfc |
| rec_conv_mechanism_v32_execution_plan.freeze.json | 7ecd40b1cdd129a5 |
| rec_conv_mechanism_v32_factorial_development.freeze.json | 52b765b41574f8e6 |
| rec_conv_mechanism_v32_factorial_validation.freeze.json | 0884d69d5f1f1dbe |
| rec_conv_mechanism_v32_final_opening.freeze.json | 039fb863781b5a39 |
| rec_conv_mechanism_v32_intervention_development.freeze.json | e22875a27cec6399 |
| rec_conv_mechanism_v32_intervention_validation.freeze.json | e01d671a517c77be |
| rec_conv_mechanism_v32_layer_map_development.freeze.json | 24fed7b452db683c |
| rec_conv_mechanism_v32_layer_map_validation.freeze.json | a05075c1a8794d63 |
| rec_conv_mechanism_v32_site_selection.freeze.json | e4e6a11b96bb53c4 |
| rec_conv_mechanism_v32_subsite_development.freeze.json | 76949769a5d19e08 |
| rec_conv_mechanism_v32_subsite_validation.freeze.json | 17212301ab07f35a |
| rec_conv_mechanism_v32_trace_development.freeze.json | 55e87449e33ac146 |
| rec_conv_mechanism_v32_trace_validation.freeze.json | e82917e0bda09106 |

The predeclared candidate is recurrent-output layer 30. Context mapping and secondary component probes have a narrower prospective status than the primary factorial and nominated-site test; this difference is preserved in adjudication. The independent final was not opened; h2/h4 not run. Raw records, hashes, tests and the integrity index are in `results/v32/processed/`, `tests/test_v32.py` and `v32_integrity_index.json`.

---

<!-- 20: V32_SCIENTIFIC_ANSWERS.md; sha256=a36651881aa33c4565f2900a1f09d94f426da8b4af7344b6e6ba2d50bb5174db -->

# Scientific Answers — V32

Answers are scoped to this hybrid model, frozen h1 six-probe signature and tested source/recipient semantics.

1. Yes: 100/50 fresh primary states pass the frozen replication gate.

2. Yes: 5/5 development and 5/5 validation families.

3. Yes in this tested design: REC-only median relative donor L2 0.997.

4. Yes: Conv-only 0.288 versus REC-only 0.997 median donor L2.

5. Yes: 100/100 and 50/50.

6. Development/validation median cosine 0.809/0.779 with lower bounds 0.759/0.723.

7. No under the per-row best scalar model: validation joint beats scalar in 1.000 of rows.

8. No under the rank-32 development-fitted fixed Procrustes model; full context-dependent rotations are not excluded.

9. Yes as an estimand description and exact-field causal error reduction, not as a localized anatomical mechanism.

10. Matched REC has lower error than sampled wrong-state REC in all 10 validation context targets; off-manifold caveat applies.

11. Matched REC beats wrong-token REC in all 10 validation context targets.

12. Matched REC beats shuffled REC in all 10 validation context targets.

13. Cross-state transfers differ, but a state-conditioned rule is not isolated.

14. Within-state secondary token contrasts differ; no stable token law is isolated.

15. Recurrent-output difference is measurable from layer 0 on the tested first probe; this is not a causal site.

16. No localized site qualifies; nominated layer 30 fails bidirectional gates.

17. Layer 30 interception removes a median 0.188 of REC benefit, below the strict joint gate.

18. Reverse insertion restores a median 0.083, below 0.50.

19. A frozen 4×4 REC×Conv group map is reported; no pair was formally qualified.

20. Same-group and cross-group scores differ descriptively; no same-layer privilege is established.

21. Cross-layer routing necessity is not established.

22. Raw gate projections change in the trace; actual transformed gates were not independently captured.

23. Diagnostic gate projection interception/reversal does not support a formal gate-mediation claim.

24. Joint pre-convolution qkv and normalized read were patched diagnostically; individual q/k/v or delta-update mediation remains untested.

25. Only post-residual layernorm output was patched diagnostically; residual-add mediation remains unresolved.

26. One nominated site fails; neither universal localization nor distributed necessity is proven.

27. Conditional response is geometrically broad in this sample: centered r90/r95/r99=61/75/92 out of at most 99.

28. No causal correction-basis reconstruction was run; no rank threshold licensed it.

29. Yes for A/B: six frozen probes and a broad response signature are used in every primary causal endpoint.

30. h2/h4 not opened because h1 local-mechanism qualification failed.

31. V32-A=True.

32. V32-B=True for exact conditional effect; no anatomical claim.

33. V32-C remains formally unconfirmed due narrower context mapping preregistration and off-manifold source concerns.

34. V32-D/E/F/G/H are not formally supported.

35. V32-I is not formally supported: tested site failure alone cannot establish distributed causation.

36. V32-J=False under the tested broader prospective panel.

37. Cross-model replication authorization=True because V32-B passes; no second-model result is claimed.

---

<!-- 21: V32_COMPLETE_REPORT.md; sha256=ae15ea02c844980ba8a0271401aa5062a29a741df852db38684cfba4c238af37 -->

# Complete Report — V32

**Mechanism of REC–Conv Conditional Correction — How Does Recurrent State Refine the Convolutional Future Handoff?** Parent `774a867eb4ffb91ef749e971fb78922ed312a285`; V32 base freeze `48a0ea5b83e8f4d1b62eca245e6766aa0d8b85d5c50c369d99c551e72175f004`.

The new 25/100/50/50-state design excluded V28–V31 formal states and froze ordinary natural token contrasts and six future probes before observing V32 causal responses. Exact REC+Conv reduces Conv donor error in 100/100 development and 50/50 validation primary rows; median reductions are 0.410/0.372, with residual-alignment cosines 0.809/0.779. Replication families passing: 5/5 development and 5/5 validation; alignment families passing: 5/5 and 5/5. REC-only remains weak. This qualifies V32-A and the precise V32-B statement that exact REC field replacement has a residual-aligned causal *conditional effect* on the tested response. It does **not** show REC carries independent future information or identify the computational mediator.

The development-only trace nominated recurrent-output layer 30. Six-probe output interception and reverse insertion each yielded 0/10 bidirectional successes in development and 0/10 in validation; exact writeback audit passed. No localized site qualifies. Matched REC beat wrong-token, wrong-state and artificial controls in 10/10 development and 10/10 validation context targets, but context mapping was frozen after primary factorial observations and wrong-state transfers may be off-manifold; V32-C remains formally unconfirmed. The 4×4 layer map and gate/qkv/residual diagnostic probes do not license specific mediation or a distributed-causation proof.

| formal outcome | supported |
|---|---|
| V32-A_REC_CORRECTION_REPLICATED | True |
| V32-B_REC_RESIDUAL_CORRECTION_CONFIRMED | True |
| V32-C_REC_CONTEXTUALIZES_CONV | False |
| V32-D_LOCALIZED_CORRECTION_SITE | False |
| V32-E_REC_GATE_MEDIATES_CONV_CORRECTION | False |
| V32-F_REC_QKV_OR_UPDATE_MEDIATION | False |
| V32-G_RESIDUAL_INTEGRATION_MEDIATION | False |
| V32-H_LAYER_PAIRED_REC_CONV_ROUTING | False |
| V32-I_DISTRIBUTED_REC_CONV_CORRECTION | False |
| V32-J_REC_CORRECTION_NOT_GENERAL | False |

The independent final stayed sealed; h2/h4 were not opened under the frozen h1 rule. Cross-model replication is *authorized for the V32-B estimand*, not performed. H2 remains true; H3, dynamic search and autonomous control remain unauthorized. Detailed methods, exclusions, raw machine records and per-report results are in the 20 supporting V32 reports and `V32_ALL_REPORTS.md`.
