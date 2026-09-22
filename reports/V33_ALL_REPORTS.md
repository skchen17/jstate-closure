# V33 All Reports

All V33 standalone reports reproduced below in order. The frozen machine records remain authoritative.

# V33 Model-2 Selection

Independent pretrained hybrid family: `tiiuae/Falcon-H1-1.5B-Base`, fixed revision `06d7330266253c20784f58b0a846dc09a9d12cee`; separate from `Qwen/Qwen3.5-4B`. The official model card and config identify Falcon-H1 as a Transformer/Mamba-2 hybrid. The local checkpoint SHA-256 is `9acd2ef3e946e88a6e8cb14d38942f2ffbf6f52a5bc37312c1e0cc64574f1aeb`; tokenizer SHA-256 `eb7825ecac026cc37e37c03d7e8d06d1f85c7ab8bdefabe81fc1b40f0ed7929a`; config SHA-256 `7ae20392a453be5f1a59b39b1e1f4fc09522901eacd64492e649672594ea4c77`. The checkpoint lives outside Git at `/data/CSK/J-space-project/models/Falcon-H1-1.5B-Base-06d7330`.

Runtime: Transformers 5.12.1, PyTorch 2.10.0+cu128, bfloat16, CUDA:0, eager attention, unfused/naive Mamba path, context cap 512. Model identity and thresholds were frozen in `aa843d414c7ad5866a47beebc665db9d0379d2a622215b1465cdc4d6ae41b600` before model-2 causal response observation. This model is architecturally independent, but its Mamba-2 update algebra is not identical to Qwen3.5 gated delta recurrence.

Official model card: https://huggingface.co/tiiuae/Falcon-H1-1.5B-Base . Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Model-2 Architecture Comparability

All 24 Falcon hybrid blocks expose four persistent tensors per layer (96 schema rows). `recurrent_states` is a Mamba-2 SSM matrix updated across tokens; `conv_states` is the depthwise short-convolution input ring; `keys` and `values` store positional attention history. The first inspected REC2/CONV2/KV shapes are `[1, 48, 64, 256]` / `[1, 3584, 4]` / `[1, 2, 51, 128]`. Mapping hash `fb4367c09e67ebf4999ce353b740c11498902c361b10d0d1b4175f48a426adaa`.

Incoming cache is captured after the shared prompt prefix. Two distinct ordinary current tokens run naturally, producing same-length outgoing caches. Current-token logits are already computed at the transplant boundary; six subsequent frozen probes measure the future. REC2 and CONV2 are exact persistent-field functional/structural analogues, not asserted to share Qwen's microscopic recurrence algebra. KV is a separate attention contrast. RoPE/position semantics are preserved by replacing only the newly appended KV slot, retaining the shared prefix.

Verdict: **comparable for the high-level V32 REC×Conv causal estimand**, not necessarily for individual q/k/v, gate, or update-term homology. `J_NOT_COMPARABLE`: Falcon has no Qwen-specific J-space target, so the architecture-independent response blocks are used. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Model-2 Native Write Interface Audit

Exact REC2, CONV2, REC2+CONV2, KV, and full-field donor transplants passed all-layer tensor equality checks. Every nonselected field remained bitwise recipient-native; shared KV prefix remained identical and only the appended slot was replaceable. The 24-layer schema records layer class, tensor shape, dtype, device, and SHA-256 for each field. The audit used one calibration prompt and two natural current tokens (`549`, `537`), without measuring future causal response.

|condition|transplanted fields|exact requested/untouched|
|---|---:|---|
|CONV2|24|True/True|
|KV|48|True/True|
|REC2|24|True/True|
|REC2+CONV2|48|True/True|
|REC2+CONV2+KV|96|True/True|
|RECIPIENT|0|True/True|

Native partial transplant—not a full-cache identity copy—is the primary mechanism test. `architecture_fields_v33.parquet`, `architecture_audit_v33.json` and frozen mapping digest make this auditable. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Prospective Replication Design

Base freeze `aa843d414c7ad5866a47beebc665db9d0379d2a622215b1465cdc4d6ae41b600`; architecture freeze `604635882bdd8ba362c532be10424af11294b4eab29b9c700beb5b4d2f9e2b47`; design freeze `62ff37b529c343238597d9705adbf6ee92487966ed0f37995cb70e95541f2ad0`. Model-2 roles: 25 calibration / 100 development / 50 validation / 50 independent final, balanced over boolean logic, modular arithmetic, graph traversal, state transition and variable binding. State IDs and prompt meanings follow frozen V32 response-blind panels; model-2 token IDs, a 40-token four-class natural library, and six prefix-probability probes were selected independently. No current-token write geometry or causal future response entered selection.

Same incoming cache forks to anchor `:` (Falcon token 549) and one deterministic eligible candidate. Y00/Y10/Y01/Y11/Ydonor use exact native fields; all primary hybrids keep recipient KV. D, C, R, RC, E_conv, R_given_C and interaction follow the fixed V32 definitions. Targets concatenate six-probe selected logits, log probabilities, signed broad logits, late hidden and four-layer workspace; no artificial J. Block scales are calibration-clean empirical RMS variability, fixed before causal writes.

Locked correction: ≥80% positive, median reduction ≥0.20, bootstrap 2.5% lower bound >0, ≥4/5 families. Locked residual alignment: median cosine ≥0.50, bootstrap lower ≥0.25, ≥4/5 families. Final opens only after development and validation pass both; Phase B uses the same opening rule. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Primary REC2×CONV2 Factorial

|role|n|positive|median reduction|reduction LB|median align|align LB|correction fam|alignment fam|
|---|---|---|---|---|---|---|---|---|
|development|100|1.000|0.388|0.351|0.793|0.766|5|5|
|validation|50|1.000|0.413|0.300|0.811|0.725|5|5|
|independent final|50|1.000|0.363|0.312|0.776|0.736|5|5|

All five native conditions were measured on each of 100 development, 50 validation and—after explicit opening—50 independent-final states, with six frozen probes each. Full tensors are in role-specific `factorial_*_v33.parquet`, `factorial_vectors_*_v33.npz` and `factorial_audit_*_v33.parquet`. Every primary hybrid retained recipient KV; writeback and per-condition field hashes were audited. Development and validation satisfy both frozen gates, and the independent final confirms direction without model/threshold adaptation. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 REC2 Conditional Correction Replication

Primary statistic `(||D-C||-||D-RC||)/||D-C||`: development 0.388, validation 0.413, independent final 0.363. Positive rows: 100/100, 50/50, 50/50. State-bootstrap lower bounds: 0.351, 0.300, 0.312. Residual-alignment cosine medians: 0.793, 0.811, 0.776; lower bounds 0.766, 0.725, 0.736. All five families pass in each role.

This is causal conditional field-replacement evidence for model 2, not a proof that REC2 carries the donor response on its own or that a particular kernel operation mediates it. V33-A **PASS**. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Conv-Dominant Handoff

|role|REC-only donor relative L2|CONV2-only donor relative L2|REC2+CONV2 relative L2|CONV/REC error ratio|
|---|---:|---:|---:|---:|
|development|0.997|0.242|0.135|0.244|
|validation|1.001|0.240|0.126|0.244|
|final|0.997|0.227|0.140|0.228|

All 5/5 families pass the prospective CONV2 dominance criterion in each role. This is an immediate one-token future readout test, not a claim that attention KV or REC2 is generally unimportant. V33-B **PASS**. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Gain and Fixed-Rotation Falsification

The descriptive per-row best scalar of CONV2 is beaten by the true joint response in 1.000 of validation rows; median scalar donor error 0.239 versus true joint 0.126. Development-only global scalar α=0.988 has validation median error 0.240, true joint better in 1.000.

Development-only rank-32 Procrustes map, identity outside fitted subspace, has validation median error 0.234; true joint is better in 1.000. This excludes this fixed low-capacity linear rotation, not every nonlinear or state-dependent map. Scalar fit is descriptive oracle; no intervention data were used to retune the primary gate. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Matched and Mismatched REC2 Context Controls

Ten predeclared development plus ten validation targets (two per family per role). Matching Conv was held fixed. Correct matched REC2 beat wrong-token, same-family wrong-state and layer-shuffled REC2 in **20/20** targets each; the all-three success fraction is 1.000. Each of five families passes in development and validation.

|condition|median donor relative L2|matched better fraction|
|---|---:|---:|
|WRONG_TOKEN_REC2|0.289|1.000|
|SAME_FAMILY_WRONG_STATE_REC2|0.293|1.000|
|SHUFFLED_REC2|1.795|1.000|
|CROSS_FAMILY_WRONG_STATE_REC2|0.481|1.000|
|RANDOM_SAME_NORM_REC2|2.492|1.000|
|MATCHED_REC2|0.157|—|
|RECIPIENT_REC2 / Conv-only|0.252|—|

Cross-state, layer-shuffle and random controls can be off-manifold; wrong-token same-prefix is the cleanest context-specific contrast. Mapping was frozen after primary factorial but before these control responses. V33-D **PASS within the frozen 20-state confirmatory subset**, with that narrower timing acknowledged. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Attention-KV Contrast

On the frozen 20-state subset, KV-only new-slot donor replacement has median donor relative L2 0.984; CONV2+KV 0.189; REC2+CONV2 with recipient-native KV 0.157. Thus the tested immediate donor future is substantially reproduced through the recurrent/Conv route while KV-only is weak for this endpoint. Older KV history remains recipient-native in all partial contrasts; this does not deny general attention memory importance. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Dimensionless Cross-Model Effect Shapes

|model / role|REC-only L2|Conv-only L2|joint L2|correction reduction|alignment|interaction ratio|
|---|---:|---:|---:|---:|---:|---:|
|Qwen3.5 / development|0.998|0.283|0.163|0.410|0.809|0.252|
|Qwen3.5 / validation|0.997|0.288|0.180|0.372|0.779|0.247|
|Falcon-H1 / development|0.997|0.242|0.135|0.388|0.793|0.224|
|Falcon-H1 / validation|1.001|0.240|0.126|0.413|0.811|0.228|
|Falcon-H1 / final|0.997|0.227|0.140|0.363|0.776|0.219|

All values are within-model dimensionless normalizations; raw hidden/logit norms are not compared across models. The structural ordering and conditional alignment replicate closely, but two models cannot establish universality. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 REC×Conv Interaction Comparison

The frozen interaction `I_RC=Y11-Y10-Y01+Y00` has median `||I_RC||/||D||` 0.224 development, 0.228 validation, 0.219 independent final. V32 Qwen3.5 median interaction was 0.252 development and 0.247 validation. This is material relative to the prospective ≥0.050 threshold, consistent with conditional rather than simply independent/additive channels. It is not a localization claim. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Replication Adjudication

Architecture comparable; locked development and validation correction/alignment gates pass; final confirms; Conv dominance, gain falsification, material interaction, and matched-context control pass. Level 1 qualitative, Level 2 formal, Level 3 structural: **PASS / PASS / PASS** in the tested models.

|formal outcome|supported|
|---|---|
|V33-A_CROSS_MODEL_REC_CORRECTION_REPLICATED|True|
|V33-B_CONV_DOMINANT_HANDOFF_REPLICATED|True|
|V33-C_CONDITIONAL_NOT_ADDITIVE_OR_GAIN_ONLY|True|
|V33-D_MATCHED_CONTEXT_CORRECTION_REPLICATED|True|
|V33-E_CROSS_MODEL_CHANNEL_ORGANIZATION_PARTIAL|False|
|V33-F_ARCHITECTURE_SEMANTICS_NOT_COMPARABLE|False|
|V33-G_MODEL_SPECIFIC_REC_CORRECTION|False|
|V33-H_SHARED_MICRO_MEDIATOR_IDENTIFIED|False|
|V33-I_SHARED_EFFECT_WITH_DIFFERENT_MICRO_MEDIATORS|False|

The predeclared failure branches (channel semantics differ, handoff-only, alternate carrier, no analogous organization) are not selected. V33-G is false for this model pair; failure was not observed. H/I are **not established**, rather than evidence of absent or different micro-mediators, because bidirectional mediation was not performed. Cross-model replication is not universality. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Phase-B Authorization and Scope

The frozen rule opened Phase B only after development and validation each passed correction and alignment. Both did, so Phase B is **authorized**. The model-2 independent final was separately opened under the same locked rule; sealed V32 model-1 final was not reopened.

A one-state-per-model, 24-layer instrumentation pilot captured exact kernel-consumed gates and true update terms; it did **not** perform multi-state bidirectional remove/restore validation. Therefore V33-H and V33-I are not licensed. Full Phase-B mediation remains a future experiment even though instrumentation is available. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Transformed-Gate Instrumentation / Mediation

Qwen3.5: raw a/b projections were independently hooked and recomputed transformed g/beta matched the actual recurrent-kernel inputs exactly in 24/24 linear-attention layers for one future probe. Falcon-H1: transformed dt/decay were reconstructed from the actual in-projection and post-Conv factors, with exact state-update reconstruction in 24/24 layers. These are architecture-specific gate/decay operations, not identical algebra.

No gate-specific removal/reverse transplant across development and validation has been executed. **Gate mediation: NOT ESTABLISHED.** Raw projection alone was not used as a mediation claim. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Post-Conv Factor Capture / Mediation

Qwen3.5: the actual cached convolution-update output and q/k/v consumed by the recurrent kernel were captured and hashed for 24 layers, rather than substituted with pre-convolution projections. Falcon-H1: the actual post-convolution x/B/C Mamba factors were captured in 24 layers. These are analogous post-Conv recurrence inputs but not identical q/k/v anatomy.

No writable post-Conv factor removal and reverse restoration was validated in both models. **Shared q/k/v mediation: NOT ESTABLISHED.** Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 True Recurrent-Update Term Capture / Mediation

Qwen3.5: exact one-token torch kernel recurrence was replayed from captured consumed q/k/v, transformed g/beta and initial state; `delta` and rank-one write matched outgoing kernel state in 24/24 layers. Falcon-H1: post-Conv x/B plus transformed dt/decay reconstructed the actual dBx update and outgoing state argument bitwise in 24/24 layers. Per-layer hashes are in `phase_b_instrumentation_v33.json`.

The update terms are genuinely consumed/reconstructed, not inferred from raw projection only. However neither update term was intercepted bidirectionally to test ≥50% benefit removal/restoration. **Update-term mediation: NOT ESTABLISHED.** Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Recurrent Read Mediation

Qwen3.5's actual recurrent-kernel output was captured during the Phase-B pilot. Falcon-H1's recurrent module emits an SSM read after x/B/C recurrence, but no matched, writable read-site intervention with equality audit was run. The site category is anatomically plausible, not confirmed as a shared mediator. **NOT TESTED BIDIRECTIONALLY**. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Residual-Integration Mediation

The two decoders both combine a recurrent/mixer result with residual and attention/MLP streams, but their layouts differ. No exact, homologous residual-integration insertion/reversal satisfying the frozen ≥50% and cosine≥0.80 criterion was run. **NOT ESTABLISHED**; V32's failed single-site result remains unchanged. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Cross-Model Micro-Mediator Comparison

High-level REC-like + Conv-like conditional correction is replicated. Exact kernel-input pilot shows model-specific anatomy: Qwen gated-delta q/k/v, transformed g/beta and rank-one delta versus Falcon Mamba-2 post-Conv x/B/C, dt/decay and dBx. Equality/audit alone does not test necessity or sufficiency. Without bidirectional development/validation interventions, neither a shared micro-mediator (V33-H) nor different micro-mediators supporting the same effect (V33-I) is established.

Phase B remains authorized for a separately frozen intervention program. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Strict Interface and Claim Audit

- Model-2 family independent; checkpoint/tokenizer/config SHA-256 frozen before outcome observation.
- REC2/CONV2 are persistent exact native cache fields across all 24 layers. Tensor schema, positions and donor/recipient equality audited.
- Primary four hybrids retain recipient KV. Full-donor is a ceiling, not a mechanism demonstration.
- 25/100/50/50 state roles disjoint; model-2 tokens and six probes selected response-blind; no J-space fabrication.
- Dev/validation/final gates were unchanged; final opened only after both earlier roles passed; model-1 V32 final stayed sealed.
- Context mapping froze after primary outcomes but before context responses; cross-state controls may be off-manifold.
- Phase-B pilot verifies actual kernel terms, but no remove/restore mediation; H/I withheld.
- No universal mechanism, application benchmark benefit, compact state, write-language or primitive-grammar claim.

Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Execution Manifest

Parent `09a00a67e5a286a05e47730c43586bce94b8095a`; base `aa843d414c7ad5866a47beebc665db9d0379d2a622215b1465cdc4d6ae41b600`. Fixed Falcon revision `06d7330266253c20784f58b0a846dc09a9d12cee`; bfloat16/CUDA:0/eager attention, Transformers 5.12.1. Panels 25/100/50/50; final opened only after development and validation gate pass. Primary factorial 100/50/50 rows; context/KV 10+10 states; Phase-B exact instrumentation pilot one state per model. V33 tests: 6 passed.

|freeze artifact|digest prefix|
|---|---|
|cross_model_rec_conv_v33.freeze.json|aa843d414c7ad586|
|cross_model_rec_conv_v33_adjudication.freeze.json|7d9dd78515dca769|
|cross_model_rec_conv_v33_architecture.freeze.json|604635882bdd8ba3|
|cross_model_rec_conv_v33_context_plan.freeze.json|34a564cc5ef1b4ed|
|cross_model_rec_conv_v33_controls.freeze.json|8ea9be7434e01470|
|cross_model_rec_conv_v33_design.freeze.json|62ff37b529c34323|
|cross_model_rec_conv_v33_factorial_development.freeze.json|3a3e637948f38d2d|
|cross_model_rec_conv_v33_factorial_independent_final.freeze.json|002be259471b5088|
|cross_model_rec_conv_v33_factorial_validation.freeze.json|1fe59f76b02bc216|
|cross_model_rec_conv_v33_final_opening.freeze.json|1b699a0a6ebc7224|
|cross_model_rec_conv_v33_phase_b_instrumentation.freeze.json|0e52dd87f85d0f58|
|cross_model_rec_conv_v33_primary_adjudication.freeze.json|52e36526582587c8|

Integrity index: `results/v33/processed/v33_integrity_index.json`, SHA-256 `8ec97d6e4e87d25ac38bc322395910e80a71b89476c8ac242bb09e79d99184b1`; 74 indexed entries. The index intentionally excludes itself, this manifest and the combined report to avoid circular digests. Model weights are outside Git; checkpoint/tokenizer/config hashes are frozen. Phase-B bidirectional mediation not completed; V33-H/I not established.

Full-suite check: **298 passed, 5 failed, 3 warnings**. All five failures are historical V14/V16/V25/V28/V29 integrity tests that pin an older byte hash of the append-only `reports/FINAL_REPORT.md`; they also failed on the V32 parent and are not V33 causal/runtime failures. The new V33 tests pass 6/6.


---

# V33 Scientific Answers

1. Yes: 24 hybrid layers expose persistent REC2/CONV2/KV, with distinct Mamba-2 micro-algebra.
2. Yes: recurrent_states and conv_states are separable persistent native tensors.
3. Yes: exact donor field replacement and untouched recipient field equality passed.
4. Yes: Conv-only relative L2 ≈0.24 versus REC-only ≈1.00 in validation.
5. Yes: joint relative L2 ≈0.126 versus Conv-only ≈0.240 in validation.
6. 100/100 development, 50/50 validation, 50/50 opened final.
7. Development 0.388, validation 0.413, final 0.363.
8. Yes: validation median 0.811, lower bound 0.725.
9. Yes: 5/5 correction and alignment families in both formal roles.
10. Yes: REC-only donor relative L2 ≈1.00 versus Conv-only ≈0.24.
11. No: true joint beats per-row best scalar in all validation rows.
12. No for tested rank-32 fixed map: true joint better in 1.000 validation rows.
13. Yes: validation median interaction ratio 0.228.
14. Yes: 20/20 frozen context targets.
15. Yes: 20/20; cross-state off-manifold caveat.
16. Yes: 20/20 versus shuffled and random; artificial controls are off-manifold.
17. Yes: all primary factorials used recipient-native KV.
18. No: KV-only median relative L2 0.984 on 20 controls.
19. Yes.
20. Yes.
21. Yes for tested scalar/additive distinction; not a universal nonlinear exclusion.
22. Yes, in frozen 20-target subset, with prospective-timing caveat.
23. No: tested high-level organization is shared.
24. No: architecture-valid interface passed.
25. No: this comparable tested model replicates; specificity not supported.
26. Yes: formal dev+validation passed.
27. Qwen transformed g/beta matched consumed kernel inputs in 24/24; Falcon transformed dt/decay reconstruction audited.
28. Qwen post-Conv q/k/v captured; Falcon post-Conv x/B/C captured, not identical q/k/v.
29. Yes, Qwen delta and Falcon dBx reconstructed against true state arguments in 24/24 layers each.
30. Not tested: no bidirectional ≥50% mediation intervention.
31. Not established.
32. Not established; anatomical differences alone do not show different mediators.
33. Yes for broader replication, not universal claims.
34. Channel-aware application experiments are justified as future tests; no benefit is claimed.

Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.


---

# V33 Complete Report

**Cross-Model Replication of REC–Conv Conditional Correction** — *Is Conv-Dominant Handoff + REC Residual Correction a General Hybrid-LM Mechanism?*

Independent model: `tiiuae/Falcon-H1-1.5B-Base`, revision `06d7330266253c20784f58b0a846dc09a9d12cee`, checkpoint hash `9acd2ef3e946e88a6e8cb14d38942f2ffbf6f52a5bc37312c1e0cc64574f1aeb`. All 24 layers expose exact persistent Mamba-2 REC2, short-Conv CONV2 and attention KV fields. The high-level field transaction is comparable to V32, but microscopic update algebra differs. Model choice, architecture map, 25/100/50/50 roles, ordinary token rule, six-probe response bundle and gates were frozen before model-2 causal outcomes.

|role|n|positive|median reduction|reduction LB|median align|align LB|correction fam|alignment fam|
|---|---|---|---|---|---|---|---|---|
|development|100|1.000|0.388|0.351|0.793|0.766|5|5|
|validation|50|1.000|0.413|0.300|0.811|0.725|5|5|
|independent final|50|1.000|0.363|0.312|0.776|0.736|5|5|

Conv-only/REC-only/joint validation donor relative L2 are 0.240/1.001/0.126; exact joint beats the descriptive best scalar in every validation row. Development-only fixed rank-32 rotation is beaten by true joint in 1.000 of validation rows. Matched REC2 beats the three frozen primary mismatch controls in 20/20 states; same/cross-state artificial controls carry off-manifold caveats. KV-only median relative error 0.984 on the 20-state contrast subset. Interaction ratios are 0.224/0.228/0.219.

Formal V33 A/B/C/D **pass**. E/F/G do not apply to the observed result. Development and validation passed the locked correction/alignment gates; only then was the 50-state model-2 final opened, and it confirmed. The sealed V32 model-1 final was not reopened. This supports cross-model replication in **the two tested hybrid LMs**, not universality.

Phase B was authorized. One-state-per-model kernel instrumentation audited exact transformed Qwen gates and true delta, and Falcon post-Conv factors and true dBx in 24/24 layers each. Bidirectional multi-state mediation was not executed; V33-H/I remain **not established**, not falsified. Applications remain unbenchmarked. H2 remains; H3, dynamic-state search, and autonomous controller remain unauthorized.

Parent `09a00a67e5a286a05e47730c43586bce94b8095a`; V33 base freeze `aa843d414c7ad5866a47beebc665db9d0379d2a622215b1465cdc4d6ae41b600`; final adjudication freeze `7d9dd78515dca769d3af7c4542537ee3e8077ea802d91e6670e53ef0d5343869`. See per-topic V33 reports and the machine integrity index. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.
