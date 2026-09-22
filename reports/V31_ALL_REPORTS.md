# V31 — All Reports in One File

The 24 standalone V31 reports are reproduced below in frozen order. The individual files are authoritative.

---

<!-- 01: V31_TOKEN_LIBRARY.md; sha256=39d167493b6836273371cd9f8dd900968bd0b70966136c63bbe59b67f118a051 -->

# Token Library — V31

Response-blind 25-state calibration excluded all V28/V29/V30 formal IDs. Common candidate tokens at rank ≤8192: 949. The frozen anchor is ID 25 (`:`); 192 anchor→candidate pairs are split 160/16/16 TRAIN/VALIDATION/FINAL. Frozen surface-composition triples are 14/16/16; validation/final AB token IDs never enter fitting, while each A and B constituent does. All 255 states, six prewrite future probes and eligibility hashes were frozen before current-token writes.

| token role | frozen category | count |
|---|---|---|
| TOKEN_FINAL | CJK_SURFACE_UNRESOLVED | 3 |
| TOKEN_FINAL | FUNCTION_WORD | 2 |
| TOKEN_FINAL | LEXICAL_SURFACE | 2 |
| TOKEN_FINAL | PUNCTUATION_OR_STRUCTURE | 1 |
| TOKEN_FINAL | SHORT_ALPHA_AMBIGUOUS | 1 |
| TOKEN_FINAL | UNCLASSIFIED | 7 |
| TOKEN_TRAIN | CJK_SURFACE_UNRESOLVED | 13 |
| TOKEN_TRAIN | FUNCTION_WORD | 22 |
| TOKEN_TRAIN | LEXICAL_SURFACE | 37 |
| TOKEN_TRAIN | NUMERIC | 10 |
| TOKEN_TRAIN | PUNCTUATION_OR_STRUCTURE | 57 |
| TOKEN_TRAIN | SHORT_ALPHA_AMBIGUOUS | 14 |
| TOKEN_TRAIN | UNCLASSIFIED | 7 |
| TOKEN_VALIDATION | CJK_SURFACE_UNRESOLVED | 1 |
| TOKEN_VALIDATION | LEXICAL_SURFACE | 4 |
| TOKEN_VALIDATION | PUNCTUATION_OR_STRUCTURE | 2 |
| TOKEN_VALIDATION | UNCLASSIFIED | 9 |

The AB relation is *single-token decoded surface concatenation*, not a proven semantic/function composition. Many examples are punctuation or word fragments (e.g. `-`+`based`); ambiguous categories remain explicitly unresolved. This limits generalization beyond the tested surface proxy. No validation-composition or TOKEN_FINAL write was opened. Machine source: `results/v31/processed/design_v31.json`, hash `2edd84b679a44578603125971abe78f5c85643be013ee2a174085f1982ba3ac1`.

---

<!-- 02: V31_WRITE_PRIMITIVE_GEOMETRY.md; sha256=06824e7fc5275a80b396e27eba5b1d2b060728727d6cbf69d25e6aae92cb27d8 -->

# Write Primitive Geometry — V31

The exact architecture-valid REC+Conv contrast has 13,369,344 components. The frozen 100 development fit states × 10 TRAIN contrasts produce 1000 rows. Pooled descriptive r90/r95/r99 = 137/220/527; this is neither a model-state dimension nor a count of causally reusable primitives.

| TRAIN-token prefix | rows | r90 | r95 | r99 |
|---|---|---|---|---|
| 128 | 808 | 114 | 184 | 433 |
| 16 | 112 | 23 | 39 | 71 |
| 160 | 1000 | 137 | 220 | 527 |
| 32 | 224 | 39 | 65 | 133 |
| 64 | 424 | 67 | 109 | 239 |

Rank grows with token coverage. This alone cannot distinguish reusable composition from continually new causal directions. Full tensor matrix and bases stay off Git under `/data/CSK/J-space-project/v31-write-work`; spectrum and hashes are committed.

---

<!-- 03: V31_PRIMITIVE_DICTIONARIES.md; sha256=2743f06fc9e0c2c301762fcc4e5ddcc7feca1f275c8742747f24b95f566a9cec -->

# Primitive Dictionaries — V31

P0 PCA, P1 TRAIN-only sparse dictionary, P2 TRAIN-only clustered prototypes, P3 frozen-surface-category PCA, and P4 TRAIN-only six-probe REC+Conv response factor were fit before held-out causal tests. P5 Conv-depth profiles and P6 exact REC-conditional Conv factorial are separately recorded as mechanisms, not deceptively counted as learned 128-atom dictionaries. M={8,16,32,64,128}, active s={1,2,4,8,16} when estimable. No deep autoencoder was introduced.

| candidate | fit span/rows | maximum estimable M/ranks | hash prefix |
|---|---|---|---|
| PCA baseline | 256 | 128 | 39c31ff06265fe01 |
| sparse dictionary | 256 | 128 | 02dcb023e7fc90c4 |
| clustered prototypes | 256 | 128 | 1ba6eb3cc9a0a0eb |
| function-conditioned | category dependent | CJK_SURFACE_UNRESOLVED:82, FUNCTION_WORD:137, LEXICAL_SURFACE:229, NUMERIC:59, PUNCTUATION_OR_STRUCTURE:256, SHORT_ALPHA_AMBIGUOUS:84, UNCLASSIFIED:41 | e4a1bd937bfc2036 |
| response factor | 200 | 199 | 127db59e1c6f0051 |

Category `UNCLASSIFIED` has only rank 41; its M64/M128 rows are explicitly not estimable. These are candidate directions, not causally validated primitives.

---

<!-- 04: V31_PRIMITIVE_CAUSAL_REALIZATION.md; sha256=9b989c04c4f7059e88c8fa7244ff2dee66ab6d51318dabae575bbfbbdc89ca13 -->

# Primitive Causal Realization — V31

All 40 development held-out AB/state pairs used the exact V29/V30 native writeback interface and six fixed future probes. Dictionary coefficients are oracle projections of the *held-out natural write* onto TRAIN-only atoms: this is a compression/reconstruction ceiling, **not** prediction of AB from A/B. Reconstruction tensor L2 is secondary; future-response cosine, magnitude and relative L2 decide.

| condition | rows | median cosine | median magnitude | median causal L2 | gate pass |
|---|---|---|---|---|---|
| EXACT_REC_CONV | 40 | 0.967 | 0.975 | 0.255 | False |
| PCA_M32 | 40 | 0.839 | 0.878 | 0.547 | False |
| PCA_M64 | 40 | 0.856 | 0.872 | 0.519 | False |
| PCA_M128 | 40 | 0.872 | 0.899 | 0.494 | False |
| SPARSE_DICTIONARY_M128_s4 | 40 | 0.831 | 0.876 | 0.562 | False |
| SPARSE_DICTIONARY_M128_s16 | 40 | 0.859 | 0.896 | 0.518 | False |
| CLUSTERED_PROTOTYPES_M128_s4 | 40 | 0.826 | 0.852 | 0.566 | False |
| FUNCTION_CONDITIONED_M32 | 40 | 0.817 | 0.869 | 0.581 | False |
| RESPONSE_FACTOR_M128 | 40 | 0.826 | 0.856 | 0.565 | False |

Passing non-ceiling development candidates: `[]`. The full machine table contains every estimable M/s condition and unavailable rows. Per-family ≥4/5 and per-row gates were applied without retuning. Because no finalist qualified, the frozen primitive validation/final panels stayed sealed.

---

<!-- 05: V31_HELDOUT_COMPOSITION.md; sha256=3ba1e1728b2a6ef81ca40a0496c2f6386e7a1c7a6831898d28c60d8a505eb347 -->

# Held-Out Composition — V31

The primary test held out the AB *single token* while its A and B constituents were individually in TRAIN. All three forks start from the same prefix cache. The 20 unseen development states × two frozen AB combinations yield 40 tests; no a→b+b→c tensor identity was used.

| condition | n | cosine | magnitude | causal L2 | success fraction | pass |
|---|---|---|---|---|---|---|
| EXACT_REC_CONV | 40 | 0.967 | 0.975 | 0.255 | 0.65 | False |
| UNIT_ADDITIVE | 40 | 0.767 | 1.093 | 0.731 | 0.0 | False |
| GLOBAL_SCALAR_GATED | 40 | 0.751 | 0.776 | 0.662 | 0.0 | False |
| LOW_ORDER_INTERACTION | 40 | 0.752 | 0.777 | 0.662 | 0.0 | False |
| STATE_CONDITIONED_SCALAR_GATED | 40 | 0.746 | 0.760 | 0.667 | 0.0 | False |
| A_ONLY | 40 | 0.535 | 0.789 | 0.908 | 0.0 | False |
| B_ONLY | 40 | 0.772 | 0.889 | 0.650 | 0.0 | False |
| SIGN_FLIPPED_B | 40 | -0.015 | 0.475 | 1.123 | 0.0 | False |

No additive or low-order composition passes. Even the exact REC+Conv donor-field transplant misses the strict four-of-five-family development gate (3/5); this ceiling is not relabeled a compositional model. Frozen selection disallows trying new formulas on validation or independent final. This is failure of the *tested surface-composition proxy*, not proof that meaningful semantic primitives do not exist.

---

<!-- 06: V31_NONLINEAR_COMPOSITION.md; sha256=dcee5b9840eceaa6568ddc562101ff285bec1d87fe30c583c35583ef9ad2340f -->

# Nonlinear Composition — V31

TRAIN-only global scalar coefficients are `[0.378449, 0.546573]`. The predeclared low-order bilinear formula has coefficients `[0.378461, 0.546551, 0.053973]`; the interaction coefficient is not selected from held-out outcomes. There were 1382 eligible TRAIN triplets across 100 fit states; training median additive *tensor* L2 = 1.023.

Held-out causal L2 is 0.731 additive, 0.662 globally gated, 0.662 low-order interaction and 0.667 state-conditioned gating. None reaches ≤0.30 or ≥4/5 families. Thus V31-C is not established; no large memorizing model was fit.

---

<!-- 07: V31_PRIMITIVE_REUSE.md; sha256=dfb415e622d12a6963e271a6db02c3029e78ebe4f76aa80251d8d60b6a5425ca -->

# Primitive Reuse — V31

For M128/s4 oracle encodings, reuse is counted only when an atom appears with ≥2 token IDs, ≥2 states and ≥2 task families.

| candidate | used atoms | geometric multi-token/state/family atoms | max families/atom | causal gate |
|---|---|---|---|---|
| SPARSE_DICTIONARY | 50 | 15 | 5 | False |
| CLUSTERED_PROTOTYPES | 53 | 12 | 5 | False |

A repeated OMP index is **geometric reuse only**. Since the held-out causal gate fails, no index is promoted to a reusable causal primitive. No semantic label is assigned to PCA axes or clusters.

---

<!-- 08: V31_PRIMITIVE_EFFECT_SIGNATURES.md; sha256=42d7bbb61c269d4b7e6e437f0da6d6c50b8ab65336dab6fc820406373933dd8e -->

# Primitive Effect Signatures — V31

The P4 response-factor directions were trained from 200 TRAIN-only exact REC+Conv transplants on six frozen probes; rank 199. Their held-out causal realization is in `primitive_causal_development_v31.parquet`. Isolated atom effects were not promoted to stable functional identities, because no dictionary passes the frozen causal gate. A TRAIN covariance or shared tensor direction alone is insufficient to claim token/state/family-stable effects. No isolated-effect transfer to validation/final was opened.

---

<!-- 09: V31_REC_CONV_INTERACTION.md; sha256=4a1a888365d6a252810c79dc639131b10356d3205176af7999cd2cf304ca05fe -->

# REC–Conv Interaction — V31

Exact Y00 recipient, Y10 REC-only, Y01 Conv-only and Y11 REC+Conv donor fields were transplanted on the same background with recipient KV untouched. Interaction I=Y11−Y10−Y01+Y00; no additivity is assumed.

| role | pairs | median ‖I‖/‖donor move‖ | median cos(I, donor) | REC gain ratio | joint direction improvement |
|---|---|---|---|---|---|
| development | 20 | 0.244 | -0.014 | 1.001 | 0.015 |
| validation | 20 | 0.244 | -0.002 | 1.001 | 0.017 |

Interaction ratio around one quarter indicates conditionality. Gain ratio near one argues against pure scalar gain; small direction improvement and residual alignment support a corrective component. This is confined to the six-probe h1 endpoint.

---

<!-- 10: V31_REC_CONV_RESIDUAL_CORRECTION.md; sha256=5c5e16cb478d0fa44a25e24f3f99ac7a0f276f6ea32a861d5c14ea9f9cddcc70 -->

# REC–Conv Residual Correction — V31

For E=Y_donor−Y_Conv and ΔREC|Conv=Y_REC+Conv−Y_Conv, the exact paired outcomes are:

| role | pairs | REC lowers L2 | Conv L2 | joint L2 | REC-only L2 | cos(ΔREC,E) | projection fraction |
|---|---|---|---|---|---|---|---|
| development | 20 | 20 | 0.345 | 0.256 | 1.004 | 0.595 | 0.360 |
| validation | 20 | 20 | 0.321 | 0.239 | 1.002 | 0.674 | 0.450 |

All 40/40 paired errors decrease and residual alignment is positive in both roles. This independently strengthens V30's narrower conditional-REC observation, but REC-only remains weak. The frozen full-donor development family gate is only 3/5; a new post-hoc threshold for a formal V31-F finalist was **not** inserted, so independent final remains sealed. No independent REC carrier is claimed.

---

<!-- 11: V31_REC_CONV_STATE_TOKEN_DEPENDENCE.md; sha256=d012daa5a88e2f7066cd7f489efe1877cbef1144169ef00671bc08dfc1792e8e -->

# REC–Conv State/Token Dependence — V31

The REC correction varies across prewrite token categories and families; sample sizes are small and mostly unresolved surface fragments.

| frozen token category | pairs | median L2 improvement | median residual alignment | median interaction |
|---|---|---|---|---|
| CJK_SURFACE_UNRESOLVED | 3 | 0.171 | 0.871 | 0.298 |
| LEXICAL_SURFACE | 8 | 0.030 | 0.387 | 0.184 |
| PUNCTUATION_OR_STRUCTURE | 4 | 0.051 | 0.511 | 0.259 |
| UNCLASSIFIED | 25 | 0.072 | 0.675 | 0.245 |

This is descriptive heterogeneity, not a verified category-specific gating law. The six-probe endpoint cannot identify REC semantic content or prove a context-invariant mechanism.

---

<!-- 12: V31_CONV_DEPTH_PATTERNS.md; sha256=a8d614e1f5669c86c8523dacc58389e74624e12b035fa259f80fe6dd9d3ccab5 -->

# Conv Depth Patterns — V31

TRAIN-only native write contrasts yield 24,000 Conv-layer energy records (1000 writes × 24 layers). Median relative energy by layer:

| Conv layer | median energy fraction | p90 fraction |
|---|---|---|
| 0 | 0.129 | 0.170 |
| 1 | 0.015 | 0.018 |
| 2 | 0.018 | 0.023 |
| 4 | 0.028 | 0.036 |
| 5 | 0.038 | 0.045 |
| 6 | 0.029 | 0.036 |
| 8 | 0.029 | 0.037 |
| 9 | 0.031 | 0.039 |
| 10 | 0.026 | 0.036 |
| 12 | 0.028 | 0.035 |
| 13 | 0.029 | 0.038 |
| 14 | 0.029 | 0.040 |
| 16 | 0.027 | 0.034 |
| 17 | 0.036 | 0.045 |
| 18 | 0.030 | 0.038 |
| 20 | 0.053 | 0.062 |
| 21 | 0.049 | 0.057 |
| 22 | 0.043 | 0.050 |
| 24 | 0.050 | 0.059 |
| 25 | 0.060 | 0.073 |
| 26 | 0.048 | 0.058 |
| 28 | 0.071 | 0.082 |
| 29 | 0.040 | 0.049 |
| 30 | 0.055 | 0.065 |

This norm profile is not a causal layer carrier. Separately, development and validation exact partial-transplant tables include single layers, quartiles, halves, prefixes, suffixes and leave-quartile-out controls. Neither development selected a passing small route nor did the fallback full24 meet the development gate.

---

<!-- 13: V31_TOKEN_CONDITIONED_DEPTH_ROUTES.md; sha256=0584ea268051ecdf39c5d31675160185fb54c859ad574ec466382e5d12a0ab63 -->

# Token-Conditioned Depth Routes — V31

Frozen development-only route rule selected `FULL_CONV` as a **fallback**, not a passing route. All 24 Conv layers were retained.

| token category | development pairs | full-Conv L2 | best descriptive group | group L2 |
|---|---|---|---|---|
| CJK_SURFACE_UNRESOLVED | 1 | 0.345 | PREFIX_03 | 0.345 |
| LEXICAL_SURFACE | 6 | 0.393 | PREFIX_03 | 0.393 |
| PUNCTUATION_OR_STRUCTURE | 1 | 0.344 | PREFIX_03 | 0.344 |
| UNCLASSIFIED | 12 | 0.310 | PREFIX_03 | 0.310 |

Per-category minima are exploratory and were not promoted to validation after selection; sparse categories and failure of the full-Conv development gate preclude a token-conditioned depth-route claim. TRAIN energy differences do not substitute for predicted-group versus wrong/random matched-group causal superiority. V31-H is not established.

---

<!-- 14: V31_CROSS_TOKEN_PRIMITIVE_TRANSFER.md; sha256=2ee5ed3e6fcaa92f61ad42bbdd01c366cbb4a09ae5bd37e6af7ee71a82bfefac -->

# Cross-Token Primitive Transfer — V31

All dictionary atoms are TRAIN-only, and development held-out AB IDs are token-OOD. Their oracle coefficients and exact writebacks were measured in the primitive-causal grid. However, using the held-out natural write to choose coefficients is **not** prediction or transplantation of a preidentified causal atom across tokens. No model passed the frozen development causal gate; an isolated cross-token primitive transfer finalist was therefore not taken to validation/final. Geometric index recurrence is not called causal transfer.

---

<!-- 15: V31_CROSS_STATE_PRIMITIVE_TRANSFER.md; sha256=d44b02d8ed98d82f41a570bc04ac7bba6380748e6cde39c61bc2fce28239cb6f -->

# Cross-State Primitive Transfer — V31

The 20 development holdout states are disjoint from all 100 dictionary-fit states and all V28/V29/V30 formal panels. Candidate atoms were applied through native target-state writeback and judged on six-probe responses. No candidate passed the causal/family gate, so there is no validated state-transportable primitive. Exact native donor-field transplants are controls, not evidence of a transferable learned primitive. Independent final was not opened.

---

<!-- 16: V31_EFFECT_VS_TENSOR_PRIMITIVES.md; sha256=9322bb0e481920e9ee0d240e5b490caa81ae6cc9f4cc23c21ff18cc783b3b35b -->

# Effect versus Tensor Primitives — V31

Tensor reconstruction and causal response were recorded on the same held-out rows.

| candidate | estimable rows | median tensor L2 | median causal L2 | Spearman across rows |
|---|---|---|---|---|
| PCA_M128 | 40 | 0.487 | 0.494 | 0.525 |
| SPARSE_DICTIONARY_M128_s16 | 40 | 0.513 | 0.518 | 0.631 |
| CLUSTERED_PROTOTYPES_M128_s16 | 40 | 0.524 | 0.525 | 0.557 |
| FUNCTION_CONDITIONED_M32 | 40 | 0.579 | 0.581 | 0.759 |
| RESPONSE_FACTOR_M128 | 40 | 0.559 | 0.565 | 0.301 |

Neither small tensor error nor positive rank correlation would by itself validate a primitive; the frozen donor-cosine/magnitude/L2/family gate is primary. These oracle projections are not held-out AB prediction.

---

<!-- 17: V31_TOKEN_LIBRARY_SCALING.md; sha256=24942734ecc335979cb5afd528d47f0611233219488347c750ab81e0984b9c46 -->

# Token-Library Scaling — V31

Frozen nested TRAIN-token prefixes 16/32/64/128/160 give pooled r95 39/65/109/184/220.

| tokens | observed writes | r95 |
|---|---|---|
| 128 | 808 | 184 |
| 16 | 112 | 39 |
| 160 | 1000 | 220 |
| 32 | 224 | 65 |
| 64 | 424 | 109 |

The chosen response-blind library has 160 TRAIN tokens plus 32 held-out AB tokens; the suggested 256-token point was not pre-registered as an executable library and has no causal result. No false 256-token extrapolation is made. Coverage changes both row count and token diversity; this is descriptive scaling.

---

<!-- 18: V31_PRIMITIVE_SATURATION.md; sha256=a13df5388ea8b1c47c4c9ff9d9db32dab620c2964ea0e0fb134631a26f3c4191 -->

# Primitive Saturation — V31

The frozen saturation criterion was <20% growth in *causally required primitive count* when token coverage doubles. No dictionary reaches the strict development causal gate, and no 256-token causal scale was tested. Therefore a causally required primitive vocabulary and its new-primitive rate are **not estimable**; V31-I and V31-J cannot be decided from PCA rank. Pooled write rank rises, but this neither proves nor refutes a potentially larger sparse reusable grammar.

---

<!-- 19: V31_MULTI_HORIZON_COMPOSITION.md; sha256=fb69d6def20435a28fe59b1ba7519b850a56ec3895b1e25b26e42db934386f2f -->

# Multi-Horizon Composition — V31

The frozen h2/h4 rule required an h1-qualified compositional mechanism. None passed the strict development causal/family gate. Consequently h2/h4 composition was not opened. V30's own exact-write horizon results remain historical and are not silently imported as V31 primitive persistence.

---

<!-- 20: V31_NATURAL_TRAJECTORY_COMPARISON.md; sha256=cdba4db020da5063b9170e42a892b6966f418eec5b39675196a3edf6784fe8ee -->

# Natural Trajectory Comparison — V31

No causally validated primitive/composition finalist was identified. An observational search for similar signatures in unconstrained natural generation was not undertaken because it would have no frozen primitive identity to compare and cannot replace a transplant test. This omission limits claims about unperturbed trajectories; no natural-use or semantic-module claim is made.

---

<!-- 21: V31_STRICT_INTERFACE_AUDIT.md; sha256=d1a6a18dc4b3df5ee5a0862076f6cd5af638f8380c032a9f32cd9a0e51636746 -->

# Strict Interface Audit — V31

All 255 prospective prefix-state IDs were response-blind frozen and exclude V28/V29/V30 formal IDs. Model-visible current-token forks share the same native incoming cache and equal cache length; donor and recipient complete token computation naturally. Exact REC, Conv and REC+Conv partial writes alter only their named native fields; recipient KV remains native. Off-manifold low-rank/dictionary `inject` clones recipient and writes only REC+Conv. Six next-token probes are prefix-logit-selected before the write and held fixed within each comparison. Incoming state, token pair, composition, probe, natural write, basis and response hashes are machine-recorded; per-atom and factorial-metric hashes are in `v31_record_hash_manifest.json`. Exact REC+Conv is a *field* ceiling, not full-cache identity. Historical reports/protocols/finals were not overwritten or reopened. The V31 50-state/TOKEN_FINAL independent final, validation composition, and validation primitive grid remain sealed. Large matrices/bases live only under `/data/CSK/J-space-project/v31-write-work` outside Git; committed hashes identify them. The earlier 256-basis whole-GPU attempt failed OOM before any fit freeze; the identical frozen rows were streamed by columns without changing method or thresholds.

---

<!-- 22: V31_EXECUTION_MANIFEST.md; sha256=d669b08d9fc0bd01f3dc2ef9328d011eada0cd3ad85036dc059142cf84475c16 -->

# Execution Manifest — V31

Parent commit `5113e800cd4f287648cd52edc3577eb77cd60776`; base freeze `3f81577cb083ffac1d19b2e312fc7652417dd7f031186ef58b38763fe2f9a2e6`. Frozen panels 25/120/60/50; 192 token pairs 160/16/16; surface-composition triples 14/16/16; six future probes. TRAIN fits: 100 states, 1000 natural write rows, 1382 eligible composition examples, 200 exact REC+Conv response-factor examples. Development tested 40 AB compositions and 40 primitive contrasts; REC/Conv factorial and depth each used 20 pairs in development and 20 in validation. No independent final opened. A whole-GPU basis attempt hit concurrent GPU OOM; same frozen data were streamed by columns, with no gate/model change.

| freeze artifact | digest prefix | file hash prefix |
|---|---|---|
| compositional_natural_writes_v31.freeze.json | 3f81577cb083ffac | 159d2731367e5bda |
| compositional_natural_writes_v31_composition_development.freeze.json | f17dd13f29494469 | 3f475202a79bca91 |
| compositional_natural_writes_v31_composition_fit.freeze.json | 85f99e81a5efd04b | 2af8ca054e5f251a |
| compositional_natural_writes_v31_design.freeze.json | 8893b8b041256e18 | 023000b5f27fa587 |
| compositional_natural_writes_v31_execution_plan.freeze.json | 94cca9880d337796 | 83bd20dbd0aacd9d |
| compositional_natural_writes_v31_final_opening.freeze.json | d72f2baa448ebb21 | 6dbad3ddf3acd786 |
| compositional_natural_writes_v31_function_conditioned_fit.freeze.json | 9561baef8769b363 | 02a92a4ed8bfd9b7 |
| compositional_natural_writes_v31_mechanism_development.freeze.json | 20ef201e51007f96 | c995a6e262286214 |
| compositional_natural_writes_v31_mechanism_validation.freeze.json | 2203a5e37c11a4ce | 05f9b54621b68399 |
| compositional_natural_writes_v31_primitive_causal_development.freeze.json | 38045a86513e87ef | 22d3fc55fbbcd5af |
| compositional_natural_writes_v31_primitive_collect.freeze.json | 64b1ba2d6f3fdc28 | d273260f6fc9f288 |
| compositional_natural_writes_v31_primitive_dictionary_fit.freeze.json | e4ed869b7fdda47b | 645afede0678fa0b |
| compositional_natural_writes_v31_primitive_geometry.freeze.json | f1908e0ae1d5a61f | 7edae60266f54872 |
| compositional_natural_writes_v31_record_hash_manifest.freeze.json | 0f0a89615a76db58 | 7b448b5bed11bd9e |
| compositional_natural_writes_v31_response_factor_fit.freeze.json | 59309876e3ef1e25 | 1f9bfa27a78f7af8 |
| compositional_natural_writes_v31_train_depth_energy.freeze.json | 86985808887d3ca1 | 1bda0be362f25c2b |

Decision file `results/v31/processed/final_opening_v31.json` SHA-256 `44d7a75f098b27b57637fb19ecae176bf65ba445d72c845e490f31c4bb924616`. Authorization remains H2 true, H3/dynamic search/autonomous controller/cross-model false. Tests are recorded in `tests/test_v31_protocol.py`; off-repository scratch is not Git-tracked.

---

<!-- 23: V31_SCIENTIFIC_ANSWERS.md; sha256=e22d8a12b6dcc0f9a68dcbc1a4739681dd36a71d360bfe1f2bca029c5008aed4 -->

# Scientific Answers — V31

These answers refer only to this hybrid model, frozen surface-token library and six-probe h1 response.

1. No tested dictionary causally reconstructs held-out writes at the frozen gate.

2. No causally sufficient primitive count is identified; M=8/16/32/64/128 were screened where estimable.

3. Sparse s=1/2/4/8/16 were screened where supported; no active count qualified causally.

4–6. Some atoms repeat geometrically across tokens, states and families, but none is validated as a causal reusable identity.

7. No: held-out causal reuse gate failed.

8. No: held-out surface AB effects were not predicted by tested A/B composition.

9. No: unit additive causal L2 median 0.731.

10. No low-order nonlinear rule passed; necessity of nonlinear composition in general is undetermined.

11–12. Token-OOD and joint state+token-OOD surface composition failed in development; validation stayed sealed under the selection rule.

13. Stable causal effect signatures despite tensor changes were not demonstrated.

14. Frozen surface categories do not establish function-specific causal primitives.

15. Primitive-count saturation is not estimable because no causal dictionary passed.

16. Yes, descriptive pooled r95 rises 39→220 across tested 16→160 TRAIN tokens.

17. Coexistence of growing rank with stable primitive vocabulary is not demonstrated.

18. Exact REC adds a positive residual-aligned correction in 40/40 development+validation pairs, but strict formal F gate was not qualified.

19. Pure gain is disfavored (gain ratio near 1); correction plus small rotation is compatible, not uniquely identified.

20. Correction varies by state/category, but sparse fragment classes prevent a confirmed token law.

21. TRAIN Conv depth energy varies; no causally validated class-specific route.

22. No predicted small depth route passed the development gate; full24 is fallback.

23–24. Cross-token and cross-state oracle dictionary tests failed; no predictive primitive transport is established.

25. A-only, B-only and sign-flipped controls fail; natural exact write is stronger, but no composition passes.

26. h2/h4 not opened because no h1 compositional mechanism qualified.

27. V31-A false/not identified.

28. V31-B/C/D/E not supported.

29. V31-F/G not formally confirmed; exact REC residual improvement replicated descriptively.

30. V31-H not supported.

31. Primitive vocabulary saturation not demonstrated.

32. Open-ended primitive complexity not demonstrated; only write-rank growth is observed.

33. Cross-model replication not authorized under the frozen qualification rule.

No compact model-state dimension, semantic atom, or general impossibility theorem is inferred.

---

<!-- 24: V31_COMPLETE_REPORT.md; sha256=c362da34525a284afca319357b4ae6c33da5128e0d7c4067b4817131a2b6a1eb -->

# Complete Report — V31

**Compositional Structure of Natural State Writes — Are High-Dimensional Future-Facing Writes Built from Reusable Causal Primitives?** Parent `5113e800cd4f287648cd52edc3577eb77cd60776`; frozen base `3f81577cb083ffac1d19b2e312fc7652417dd7f031186ef58b38763fe2f9a2e6`. Prospectively frozen 25/120/60/50 disjoint panels and 192 response-blind token pairs. Independent final remained sealed.

The 1000 TRAIN natural REC+Conv contrasts have 13,369,344 components, pooled r95=220, rising 39→65→109→184→220 over 16→32→64→128→160 TRAIN tokens. This is geometric growth, not a causal primitive count.

The decisive 40-case unseen surface-AB development test failed: additive, global scalar, low-order interaction and state-conditioned scalar causal L2 medians are 0.731/0.662/0.662/0.667; none passes the ≤0.30, cosine≥0.90, magnitude[0.8,1.2], ≥4/5-family gate. Sparse/prototype/function-conditioned/response-factor oracle reconstruction candidates likewise yielded no qualifying development primitive. This does **not** establish that semantic or other composition is impossible: the available AB triples are mostly surface fragments, and validation was sealed after development failures.

The exact REC×Conv factorial independently repeated a narrower result: REC lowers Conv donor error in 20/20 development and 20/20 validation pairs; residual alignment cosine medians are 0.595/0.674; interaction ratios 0.244/0.244. REC-only remains weak, gain ratio ≈1 and full24 Conv is a fallback, not a qualified small route. Because the frozen full-donor development gate is only 3/5 families and no F-specific numeric finalist threshold was predeclared, this replicated correction is not promoted to a formal V31-F finalist or used to open the independent final.

| formal outcome | supported |
|---|---|
| V31_A_REUSABLE_CAUSAL_WRITE_PRIMITIVES_CONFIRMED | False |
| V31_B_COMPOSITIONAL_WRITE_GENERALIZATION | False |
| V31_C_NONLINEAR_WRITE_COMPOSITION | False |
| V31_D_FUNCTIONAL_PRIMITIVES_WITH_STATE_DEPENDENT_REALIZATION | False |
| V31_E_TOKEN_FAMILY_SPECIFIC_WRITE_MODULES | False |
| V31_F_REC_CORRECTS_CONV_WRITE | False |
| V31_G_REC_ROTATES_OR_GATES_CONV_WRITE | False |
| V31_H_TOKEN_CONDITIONED_CONV_DEPTH_ROUTES | False |
| V31_I_PRIMITIVE_DICTIONARY_SATURATES | False |
| V31_J_OPEN_ENDED_WRITE_COMPLEXITY | False |
| V31_K_NO_STABLE_COMPOSITIONAL_STRUCTURE_IDENTIFIED | True |

V31-K means **no stable tested compositional structure identified**, not no structure exists. Causal primitive saturation versus open-ended primitive count remains unresolved; rank growth cannot decide it. H2 remains true; H3, dynamic state search, autonomous controller and cross-model replication remain unauthorized. Full files and hashes are in `results/v31/processed/`; `V31_ALL_REPORTS.md` reproduces each V31 report separately.
