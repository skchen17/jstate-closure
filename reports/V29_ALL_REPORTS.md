# V29 — All Reports in One File

This bundle contains all 22 frozen V29 reports verbatim, in the required report order. The individual files remain the authoritative sources.

---

<!-- 01: SAME_INCOMING_STATE_FORKS_V29.md; sha256=073556f84c03831afc77b5223da8f225e574a27dfd04be71cec180e4ba8744d7 -->

# Same Incoming State Forks — V29

Disjoint frozen panels: 25 calibration, 75 development, 50 validation, 50 independent final (10 per family). V28 selected state IDs were excluded. For each state, the same exact prefix cache was replayed with two different current tokens chosen exclusively from the pre-write top-16 next-token distribution; the shared primary next token came from the frozen V18 teacher continuation. Prefix, token, incoming/outgoing state, and probe hashes are in `design_v29.json` and per-role fork Parquet records. This is a top-plausible/distinct-surface rule, **not** a verified semantic-pair or same-class control.

All 25 calibration same-token replays were exact. The formal all-layer analysis found distinct outgoing states and median shared-next-token future Q `90.529` development and `92.302` validation; the independent final had median Q `92.745`. The sole current-token difference is causal for the naturally committed cache; future comparison holds the next token fixed.

---

<!-- 02: NATURAL_WRITE_CONTENT_AUDIT_V29.md; sha256=5c4450ab2845238345425a74776d10455b57c41234f289630eef7001f8183917 -->

# Natural Write Content Audit — V29

REC and Conv write contrasts are exact `P_out,B − P_out,A` at all 24 recurrent layers; KV is only the newly appended token slot at all 8 attention layers. Pre-existing KV slots are bitwise identical between branches. Geometry is descriptive, not a carrier claim.

| role | channel | fields | median contrast norm | nonzero fraction |
|---|---|---:|---:|---:|
| development | Conv | 1800 | 63.901 | 1.000 |
| development | KV | 1200 | 26.504 | 1.000 |
| development | REC | 1800 | 0.892 | 1.000 |
| validation | Conv | 1200 | 64.195 | 1.000 |
| validation | KV | 800 | 26.129 | 1.000 |
| validation | REC | 1200 | 0.901 | 1.000 |

Train-only centered **joint REC+Conv contrast** spectrum has r90/r95/r99 = `11/22/55` over 90 training examples, rank `89` in ambient dimension `13369344`. Channel-specific full-vector centered spectra were not computed; per-field norms must not be substituted for them. The joint rank is not a dynamical state dimension.

---

<!-- 03: FORK_FUTURE_SIGNATURE_V29.md; sha256=835e497b1d5fb1cc2c5b0b205bd1a6f5491eff36778b1ec55726da61654fac63 -->

# Fork Future Signature — V29

With the same frozen next token, the natural A/B outgoing caches produce large next-step contrasts in layer-30 J, logits, semantic log-probabilities, multi-layer workspace, broad-vocabulary projection, and late residual. The per-state vectors and endpoint-specific effects are in `transfer_vectors_*_v29.npz` and `transfer_*_v29.parquet`. Median aggregate future Q is `90.529` development and `92.302` validation. Branch future contrast is causal to the token-conditioned complete outgoing state but not by itself a localized field attribution.

---

<!-- 04: FULL_STATE_TRANSFER_CEILING_V29.md; sha256=6edb282325f8b3e7b74442dc955d7ee88ef59c87b74fcb0bba495f111f04908f -->

# Full State Transfer Ceiling — V29

For all 75 development and 50 validation forks, replacing every initialized native field across all 32 layers (24 REC/Conv; 8 KV new slots) made the recipient cache bitwise equal to the donor and reproduced its future response (relative L2 ≤ 1e-6), reciprocally. This is an **identity ceiling**, not evidence of a selective carrier. The initial calibration diagnostic wrongly reused the V28 posterior 8-layer field subset; its median relative L2 was 0.823. It was invalid as a full-cache control and remains preserved in an explicit interface amendment.

---

<!-- 05: SAME_BACKGROUND_PARTIAL_TRANSPLANT_V29.md; sha256=45d7e28265bf9253fb3e689f0298edc0cb7f524ad2d852fff0d13363626d66b6 -->

# Same-Background Partial Transplant — V29

All partial operations copy exact naturally produced donor fields into a recipient branch with the **same incoming state**. REC and Conv copy full native tensors; KV copies only the new slot. Untouched recipient fields and current readout are unchanged. Donor direction, magnitude, relative L2, reciprocal replication, and all five family gates were predeclared.

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
| REC+KV | development | 0.155 / 0.127 / 0.989 | 0.286 / 0.163 / 0.971 | False |
| REC+KV | validation | 0.121 / 0.125 / 0.993 | 0.342 / 0.175 / 0.962 | False |
| Conv+KV | development | 0.992 / 0.980 / 0.132 | 0.996 / 0.999 / 0.095 | True |
| Conv+KV | validation | 0.991 / 0.962 / 0.137 | 0.996 / 1.003 / 0.095 | True |
| REC+Conv+KV | development | 1.000 / 1.000 / 0.000 | 1.000 / 1.000 / 0.000 | True |
| REC+Conv+KV | validation | 1.000 / 1.000 / 0.000 | 1.000 / 1.000 / 0.000 | True |

A large ablation/write-block effect from V28 was not counted as transferable content. Here REC+Conv and Conv pass; REC-only and KV-only do not. The full-cache row is a ceiling.

---

<!-- 06: REC_WRITE_CONTENT_V29.md; sha256=1422d87fe5de329d81fe916c584bee038c422616a9151e1a836ad48ab509d38d -->

# REC Write Content — V29

REC-only exact donor transplant does not pass the reciprocal donor-direction gate in development or validation. Its natural tensor contrasts are real, but neither their norm nor V28 marginal write-block sensitivity licenses an independent transferable-REC-carrier claim. REC can modify the fidelity of the Conv-containing combination, without a globally additive decomposition.

| channel | role | A←B cosine / magnitude / rel-L2 | B←A cosine / magnitude / rel-L2 | reciprocal gate |
|---|---|---:|---:|---:|
| REC | development | 0.062 / 0.095 / 0.999 | 0.259 / 0.132 / 0.980 | False |
| REC | validation | -0.002 / 0.095 / 1.003 | 0.358 / 0.137 / 0.962 | False |
| REC+Conv | development | 0.998 / 0.995 / 0.067 | 0.998 / 0.997 / 0.067 | True |
| REC+Conv | validation | 0.998 / 0.996 / 0.070 | 0.998 / 0.996 / 0.067 | True |

---

<!-- 07: CONV_WRITE_CONTENT_V29.md; sha256=d72e3cef11ec662b927fe7edd725cac8e2f6a9be51832aa83d715cf73bcba7b7 -->

# Conv Write Content — V29

Conv-only native donor fields pass reciprocal donor-directed transfer in both formal roles while KV remains recipient-native. This identifies a transferable, token-conditioned recurrent-subsystem component of the immediate future response under same-background forks; it does not decode a semantic variable stored in Conv.

| channel | role | A←B cosine / magnitude / rel-L2 | B←A cosine / magnitude / rel-L2 | reciprocal gate |
|---|---|---:|---:|---:|
| Conv | development | 0.988 / 0.971 / 0.163 | 0.992 / 0.989 / 0.127 | True |
| Conv | validation | 0.987 / 0.962 / 0.175 | 0.992 / 0.993 / 0.125 | True |

---

<!-- 08: KV_WRITE_CONTENT_V29.md; sha256=100910374eb8319123a03ef064bd0d4b2eeb84e6a0569d3e49000f14dc8a57da -->

# KV Write Content — V29

The newly appended KV slots differ with current token and provide an exact history carrier, but KV-only replacement does **not** reproduce most of the donor-specific next-step response under the frozen targets and probes. This does not deny that KV records token/history information.

| channel | role | A←B cosine / magnitude / rel-L2 | B←A cosine / magnitude / rel-L2 | reciprocal gate |
|---|---|---:|---:|---:|
| KV | development | 0.082 / 0.067 / 0.997 | 0.108 / 0.067 / 0.995 | False |
| KV | validation | 0.088 / 0.067 / 0.996 | 0.093 / 0.070 / 0.996 | False |

---

<!-- 09: KV_HELD_FIXED_RECURRENT_WRITE_V29.md; sha256=4c5d626c00e49a218462497db94c25bc0361228d10e006d2c2903ff905758695 -->

# KV-Held-Fixed Recurrent Write — V29

In `REC+Conv` transfer, the eight attention layers' KV history remains **recipient-native**, including its new token slot. Yet future response moves reciprocally toward the donor in development and validation and the fixed independent final. This supports causal token-conditioned content in recurrent/conv persistent writes beyond direct KV token-history carryover. `KV` donor with recipient-native REC/Conv fails; the contrast is local to this design and endpoint.

| channel | role | A←B cosine / magnitude / rel-L2 | B←A cosine / magnitude / rel-L2 | reciprocal gate |
|---|---|---:|---:|---:|
| KV | development | 0.082 / 0.067 / 0.997 | 0.108 / 0.067 / 0.995 | False |
| KV | validation | 0.088 / 0.067 / 0.996 | 0.093 / 0.070 / 0.996 | False |
| REC+Conv | development | 0.998 / 0.995 / 0.067 | 0.998 / 0.997 / 0.067 | True |
| REC+Conv | validation | 0.998 / 0.996 / 0.070 | 0.998 / 0.996 / 0.067 | True |

---

<!-- 10: CHANNEL_WRITE_CONTENT_FACTORIAL_V29.md; sha256=2613198d06b207674b37c137207efa3312ff434b17683dbfcf52a6809248f616 -->

# Channel Write Content Factorial — V29

The seven frozen channel subsets form an exact native-field factorial: REC, Conv, KV, REC+Conv, REC+KV, Conv+KV, and complete REC+Conv+KV. The complete cache is an identity ceiling; the single Conv channel already passes. Therefore **no single channel sufficient / only combinations work** is false. Combinations improve fidelity but cannot be interpreted as additive main effects without interaction-aware analysis.

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
| REC+KV | development | 0.155 / 0.127 / 0.989 | 0.286 / 0.163 / 0.971 | False |
| REC+KV | validation | 0.121 / 0.125 / 0.993 | 0.342 / 0.175 / 0.962 | False |
| Conv+KV | development | 0.992 / 0.980 / 0.132 | 0.996 / 0.999 / 0.095 | True |
| Conv+KV | validation | 0.991 / 0.962 / 0.137 | 0.996 / 1.003 / 0.095 | True |
| REC+Conv+KV | development | 1.000 / 1.000 / 0.000 | 1.000 / 1.000 / 0.000 | True |
| REC+Conv+KV | validation | 1.000 / 1.000 / 0.000 | 1.000 / 1.000 / 0.000 | True |

---

<!-- 11: RECIPROCAL_WRITE_CONTENT_TRANSFER_V29.md; sha256=fdf93f846396fad8cec4c10b516206f83ad95b74d175032217d6bd1a88d6ee16 -->

# Reciprocal Write Content Transfer — V29

Both A←B and B←A were evaluated for every formal fork. Passing requires donor-directed cosine ≥0.80, magnitude ratio ≥0.50, at least half of states, and at least four of five families in each role. Exact requested/untouched writeback and identical next token are mandatory. The independent final was opened only for the fixed REC+Conv finalist after both role gates.

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

Independent final `50` states: REC+Conv reciprocal pass `True`; A←B cosine `0.998`, B←A cosine `0.997`.

---

<!-- 12: STATE_DEPENDENT_WRITE_CODE_V29.md; sha256=3ef9cb2b7ab25129cabf7683280428701db3a6231b4ccaf027639c9f3c0f4327 -->

# State-Dependent Write Code — V29

Two response-blind same-family state pairs per family used the same two tokens in a 2×2 factorial. REC/Conv token contrasts are geometrically fairly aligned across states, yet cross-state additive delta transfer degrades relative to the corresponding native same-state transfer. This delta injection is off-manifold and no cross-state success/degradation gate was frozen. Hence neither **global code** nor **state-dependent code** is formally confirmed.

| role | pairs | contrast cosine | cross-state delta→native cosine | cross-state delta relative L2 |
|---|---:|---:|---:|---:|
| development | 10 | 0.937 | 0.922 | 0.418 |
| validation | 10 | 0.958 | 0.948 | 0.347 |

---

<!-- 13: STATE_TOKEN_FACTORIAL_V29.md; sha256=ef8d7e21422ff1a3de7e6438a98389474d1756118f5513411f03f83e465dfbe2 -->

# State × Token Factorial — V29

Each frozen pair evaluates `(state A/B) × (token A/B)` with one shared next token. Exact state and token hashes, write-contrast geometry, response interaction norm, same-state REC+Conv reference, and wrong-state delta injection are in `state_token_factorial_*_v29.parquet`. This is a 10-pair-per-role diagnostic, not a global coordinate system proof.

| role | pairs | contrast cosine | cross-state delta→native cosine | cross-state delta relative L2 |
|---|---:|---:|---:|---:|
| development | 10 | 0.937 | 0.922 | 0.418 |
| validation | 10 | 0.958 | 0.948 | 0.347 |

---

<!-- 14: FUTURE_RESPONSE_SIGNATURE_V29.md; sha256=239c971a87821bda232fb64f601067a343c28ddf5d60aabf9c0ba17e63ffa069 -->

# Future Response Signature — V29

Ten frozen states per role used four next-token probes selected from the **prefix distribution before either write**. The concatenated functional signature tests whether a channel tracks donor response across inputs, not merely a single `z`. Complete-cache signatures are exact; REC+Conv remains strongly donor-directed; KV remains weak. The probe-set choice limits generalization beyond these tokens.

| role | channel | A←B cosine / rel-L2 | B←A cosine / rel-L2 |
|---|---|---:|---:|
| development | REC | 0.277 / 0.969 | 0.236 / 0.982 |
| development | Conv | 0.966 / 0.258 | 0.940 / 0.345 |
| development | KV | 0.227 / 0.975 | 0.210 / 0.980 |
| development | REC+Conv | 0.988 / 0.152 | 0.989 / 0.147 |
| development | REC+Conv+KV | 1.000 / 0.000 | 1.000 / 0.000 |
| validation | REC | 0.277 / 0.975 | 0.268 / 0.974 |
| validation | Conv | 0.951 / 0.307 | 0.932 / 0.363 |
| validation | KV | 0.239 / 0.976 | 0.214 / 0.978 |
| validation | REC+Conv | 0.986 / 0.167 | 0.989 / 0.149 |
| validation | REC+Conv+KV | 1.000 / 0.000 | 1.000 / 0.000 |

---

<!-- 15: WORKSPACE_VS_WRITE_CONTENT_V29.md; sha256=0b612dc5c8eb36f9fc1de0ccb1924692394d3dec7b402c8556c6554424a9100e -->

# Workspace Versus Write Content — V29

Train-only ridge models predict the h1 response contrast on 50 held-out validation states. `J_t` here is the layer-30 current workspace readout, **not** a cache field. Exact token identity/surface, J, broader current workspace, and per-field write geometry are compared diagnostically.

| diagnostic model | held-out validation R² |
|---|---:|
| M0_token_identity_and_surface | 0.417 |
| M1_current_J | 0.391 |
| M2_token_plus_J | 0.423 |
| M3_token_J_broader_current_workspace | 0.482 |
| M4_token_J_write_geometry | 0.488 |

Current J predicts coarse write-geometry features with held-out R² `0.926`. Adding write geometry to token+J improves held-out response R², but predictive gain does not prove current J causally insufficient for the full outgoing write. A high-fidelity direct J transplant/J-matched causal equivalence test was not established; V29-H is not confirmed.

---

<!-- 16: WRITE_EFFECT_REALIZATION_DIMENSION_V29.md; sha256=3ee6dfe4c8c32528c8f3a3bb9d63bb894d25fce4fed4d5a991c4f0329d24aad7 -->

# Write-Effect Realization Dimension — V29

A response-blind centered PCA basis was fitted to 90 natural REC+Conv token-write contrasts (25 calibration + 65 development), then causal donor-response reconstruction was tested on 10 held-out development and 50 validation pairs, both directions. This is an approximate **write-effect** test, not a compact persistent-state or dynamical-state test. The approximations are off-manifold and BF16-writeback audited.

| k | dev A←B / B←A rel-L2 | val A←B / B←A rel-L2 | strict gate |
|---:|---:|---:|---:|
| 4 | 0.226 / 0.271 | 0.217 / 0.203 | False |
| 8 | 0.215 / 0.241 | 0.213 / 0.182 | False |
| 16 | 0.196 / 0.215 | 0.168 / 0.163 | False |
| 32 | 0.181 / 0.157 | 0.146 / 0.133 | True |
| 64 | 0.165 / 0.147 | 0.123 / 0.116 | True |

Ambient REC+Conv write dimension `13369344`; train span rank `89`. Requested k `[128, 256]` exceed the available train span and were **not** called failures. No tested dimension is inferred beyond the strict gate.

---

<!-- 17: NATURAL_VS_RANDOM_WRITE_V29.md; sha256=47fbe19865ef82df34016c0daec2798459da6841c5ebd830e64ec56b68c88e9f -->

# Natural Versus Random Write — V29

Ten states per role compare exact native REC+Conv donor writes with random same-norm, within-field shuffled, sign-flipped, and wrong-incoming-state delta controls. Artificial controls are off-manifold and BF16-realization ratios are saved per row; they are not substitutes for native donor-field transfer. Natural writes are much more donor-faithful than the random/shuffled/sign-flipped controls. The wrong-state control can retain partial directionality, so it is not evidence of a uniquely state-specific code.

| role | condition | median donor cosine | median relative L2 |
|---|---|---:|---:|
| development | natural_REC+Conv | 0.998 | 0.064 |
| development | random_same_norm | 0.341 | 1.389 |
| development | shuffled_natural | 0.426 | 1.022 |
| development | sign_flipped | -0.172 | 1.182 |
| development | wrong_state | 0.875 | 0.507 |
| validation | natural_REC+Conv | 0.998 | 0.071 |
| validation | random_same_norm | 0.417 | 1.430 |
| validation | shuffled_natural | 0.415 | 1.223 |
| validation | sign_flipped | -0.186 | 1.116 |
| validation | wrong_state | 0.972 | 0.254 |

---

<!-- 18: WRITE_CONTENT_HORIZON_V29.md; sha256=54c0ff249f804345fff17ae609f16d912de6fdb8b1ef0f48e933305e6d8424e6 -->

# Write Content Horizon — V29

Only after h1 REC+Conv gates passed, 10 states per role were followed through h2 and h4. Every continuation token was frozen from the pre-write probe set, so this is a controlled, partly artificial continuation rather than unconstrained generation. Donor-specific influence may decay, rotate, or transform; the data do not establish an independent memory criterion.

| role | h | median donor cosine | median magnitude | median relative L2 |
|---|---:|---:|---:|---:|
| development | 1 | 0.998 | 0.997 | 0.064 |
| development | 2 | 0.996 | 0.989 | 0.090 |
| development | 4 | 0.968 | 0.978 | 0.258 |
| validation | 1 | 0.998 | 0.995 | 0.071 |
| validation | 2 | 0.996 | 0.990 | 0.088 |
| validation | 4 | 0.965 | 0.970 | 0.262 |

---

<!-- 19: STRICT_WRITE_INTERFACE_AUDIT_V29.md; sha256=9ff8d25493131de6610bfbe02af98d97ec0b2603ad8c8d19295002588db70e3e -->

# Strict Write Interface Audit — V29

- Parent `8d3954ef05957ab22f27261ea7eaa217497b2b62`; base freeze `86389aa651802da41d09c0217c9f588586eab45a7274aee4d5d1ae42c2114913`; design `e9bd1b261b49e01e391d361bc1b1b2fe35a1221cbc3323913afde1f59aae78f3`; interface amendment `40851cd00d278191045604417c0f7edb4df0417f0eea1bd0e81d3c90d49b91d0`.
- Initial calibration/V28-selected-layer diagnostic retained and **excluded** as full cache: only 6 REC/Conv + 2 attention layers; 53 transient development states were interrupted before any formal partition was saved. This exposure is disclosed, not silently recomputed.
- Formal cache coverage: all 24 REC/Conv layers and all 8 attention layers. Every requested native field exactly equals donor; every untouched field equals recipient; shared next token and current-readout capture are audited in Parquet.
- KV operation copies only the newly appended slot; earlier KV slots must already be bitwise equal. No sequence shortening or arbitrary KV subtraction.
- Full all-field transfer is an identity ceiling, not localization. Partial transfers can be off-manifold combinations even with shared incoming state; causal claims are restricted to the tested transplant semantics.
- Independent final opened only for frozen `REC+Conv`, after both dev/validation gates: `399fb32c0a9e607494b1c54a116a35ebfec4f7fde754eb5a2ebcc50d95f232b5`.
- No historical v1–v28 record was overwritten. H2 remains; H3, dynamic-state search, and autonomous controller are unauthorized.

---

<!-- 20: EXECUTION_MANIFEST_V29.md; sha256=ed512f59082839e956423fe01b4f4cc08ee0b867054790dedf2334e2c49689cd -->

# Execution Manifest — V29

- `PYTHONPATH=src python -m jclosure.protocol_v29 freeze`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.design_v29`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.forks_v29 calibration`
- `PYTHONPATH=src python -m jclosure.experiments.plan_v29`
- `PYTHONPATH=src python -m jclosure.experiments.amend_v29`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.forks_formal_v29 development`
- `PYTHONPATH=src python -m jclosure.experiments.analyze_v29 development`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.forks_formal_v29 validation`
- `PYTHONPATH=src python -m jclosure.experiments.analyze_v29 validation`
- `PYTHONPATH=src python -m jclosure.experiments.analyze_v29 final_opening`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.secondary_v29 development`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.secondary_v29 validation`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.diagnostics_v29`
- `PYTHONPATH=src python -m jclosure.experiments.realization_plan_v29`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.realization_v29`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.forks_formal_v29 independent_final`
- `PYTHONPATH=src python -m jclosure.experiments.adjudicate_v29`
- `PYTHONPATH=src python -m pytest tests/test_v29_natural_write_content.py -q`
- `PYTHONPATH=src python -m pytest -q`
- `PYTHONPATH=src python -m jclosure.reporting_v29`
- `git commit`
- `git push origin main`

V29 tests: `6 passed in 1.78s`. Full suite: `280 passed, 3 failed, 3 warnings in 52.90s`. Historical cumulative-report hash tests may fail when `FINAL_REPORT.md` is appended; these failures are reported, not suppressed.

---

<!-- 21: V29_SCIENTIFIC_ANSWERS_V29.md; sha256=8cb13e6b9a8527e3d12fe05a818fa533f505d9f7add1115c7471701f284df995 -->

# V29 Scientific Answers

1. Yes. Exact identical incoming caches with distinct preselected tokens yielded distinct natural outgoing caches in all formal states.

2. REC, Conv and newly appended KV fields changed across all initialized native layers; all 32 layers were audited.

3. Per-field REC/Conv/KV contrast norms are in the write-geometry Parquet records; channel medians are in the audit report.

4. Yes. Shared-next-token future Q medians were 90.529 development and 92.302 validation.

5. Yes. Exact complete native-cache replacement reproduced donor futures reciprocally; it is an identity ceiling.

6. REC-only did not pass the frozen reciprocal donor-direction gate.

7. Conv-only passed development and validation reciprocal donor-direction gates.

8. KV-only did not pass despite recording the new token/history slot.

9. REC+Conv passed development, validation and the one fixed independent final.

10. Yes for Conv, REC+Conv and Conv+KV; not for REC, KV or REC+KV under the frozen gates.

11. Yes. REC+Conv donor transfer is strong while KV remains recipient-native.

12. No. KV-only explains little of the donor-specific future contrast for these endpoints.

13. Not required for threshold success because Conv alone passes; REC+Conv improves donor fidelity.

14. V28 write-block necessity is compatible, but does not itself establish transferable content; V29 uses separate donor-direction tests.

15. Potentially, but not formally established: cross-state delta injection degrades while geometry remains aligned.

16. Token-conditioned REC/Conv writes show positive cross-state similarity, but not exact equality or decisive global transfer.

17. Cross-state additive write deltas transfer imperfectly and are off-manifold diagnostics.

18. No stable global write code was formally confirmed.

19. Current layer-30 J predicts coarse write geometry with held-out R² 0.926; full write content is not identified by this feature model.

20. Write features add some held-out response prediction beyond token+J, but causal workspace incompleteness is not established without a J-matched test.

21. Four frozen next-token probes define each measured functional response signature; complete-cache transfer is exact.

22. Yes. REC+Conv remains donor-directed over the concatenated four-probe signature.

23. Yes for the tested controls: exact natural transfer is more donor-faithful than random/shuffled/sign-flipped norm-related alternatives.

24. A strict write-effect dimension is reported only if a train-only rank-k approximation passes both held-out roles and families; see the dimension report.

25. Controlled h1/h2/h4 effects are recorded; the continuation is frozen, not free generation.

26. V29-A is True.

27. V29-B/C/D/E are True/True/False/False.

28. Neither state-dependent nor global write code passed a frozen cross-state gate; both remain inconclusive.

29. Yes, H2 remains.

30. No. H3, complete dynamic-state search, and autonomous controller remain unauthorized.

31. The token conditions native REC/Conv and KV outgoing updates; same-background Conv and REC+Conv fields carry transferable next-response distinctions beyond recipient-native KV. Specific semantic variables are not decoded.

---

<!-- 22: V29_COMPLETE_REPORT.md; sha256=795bd2da28bae2586895886d492c4c810d392aecd4f3511f74be5c50abb71aaf -->

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
