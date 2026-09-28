# V39 — All Reports

<!-- V39_FROZEN_STARTING_POINT.md -->

# V39 frozen starting point

V39 is a fresh experiment rooted at the pushed V38 commit
`3c33f945b22b7f66a90835861bdb829072a1ea38`. V1–V38 reports,
results, protocols, and the original V38 report are not rewritten.
The only V38-related new file is the append-only
`reports/V38_METRIC_AMENDMENT_V39.md`; the corresponding V39 audit is
`reports/V39_V38_INTERACTION_METRIC_AUDIT.md`.

The V38 audit established that its column `threeway_projection` was a
signed *length* component, `<I234,E111>/||E111||`, not the dimensionless
projection coefficient. V39 preregisters the corrected coefficient
`p=<I234,E111>/||E111||²`, fraction `f=||I234||/||E111||`, cosine, and
the identity `p=f·cos`; V38's historical gate result is not silently
relabelled as an amplification coefficient.

V39 uses the exact V38 model-specific target bundles and clean-scale
normalizers as fixed readout coordinates, with new response-blind natural
token forks and six probes for each V39 state. Falcon is the primary model;
Qwen is a comparative control. The four fresh, mutually disjoint pools are
calibration 20, development 80, validation 40, and independent final 40
states per model, each evenly distributed over five task families. The
formal generator horizon is 3 in every family and role; in particular,
V38's modular-arithmetic role shift is not repeated. All four pools were
sealed before any V39 formal intervention outcome was observed. Validation
and final are not substitutes for V18 training-source states.

Frozen artifacts and design files under `artifacts/interaction_genesis_v39*`,
`data/v39/`, and `results/v39/processed/` carry exact source hashes, pool
hashes, model weight/tokenizer/config hashes, condition mapping, endpoints,
probe IDs, thresholds, and stage order. The independent final is sealed until
the stated Falcon-specific development and validation gates, including a
frozen primitive mediator, pass. This is an opening rule, not a prediction
that they will pass.

---

<!-- V39_V38_INTERACTION_METRIC_AUDIT.md -->

# V39: V38 三阶交互指标审计与追加勘误

V38 原始代码将 `threeway_projection` 计算为 `<I234,E111>/||E111||`（本批状态没有触发分母下限）。这是有量纲的沿效应方向分量，不是无量纲投影系数 `p=<I234,E111>/||E111||²`。因此 V38 表中该数值大于 `||I234||/||E111||` 并不违反实现，但若读作投影系数则数学含义错误。

V38 的 `threeway_fraction` 与标准范数比一致；正负号不因正分母的额外归一化而改变，故其符号稳定性及原有门槛判定保持不变。V1–V38 文件未改动。

|模型|角色|n|旧有量纲分量中位数|修正 p 中位数|f 中位数|cos 中位数|符号稳定性|
|---|---|---:|---:|---:|---:|---:|---:|
|Q|development|80|0.048|0.004|0.161|0.025|0.600|
|Q|validation|40|0.096|0.009|0.184|0.056|0.650|
|F|development|80|0.822|0.092|0.602|0.177|0.963|
|F|validation|40|0.619|0.066|0.496|0.136|0.925|

逐状态验证 `p=f×cos` 与 `|p|≤f`；中位数之间不要求满足乘法恒等式。
完整逐状态表：`results/v39/processed/v38_interaction_metric_audit_v39.parquet`。
V39 后续一律用明确标注的无量纲 `p`、范数比 `f` 和余弦 `c`，不再含糊使用“projection”。

---

<!-- V39_HIGH_LEVEL_RECONFIRMATION.md -->

# V39 — High Level Reconfirmation

Development-only high-level donor-error gate: Falcon 1.000 positive, median reduction 0.4185, 5/5 families; Qwen 0.912, 0.3250, 5/5. Both pass the frozen 0.80/0.20/4-family gate. Validation remains sealed. Machine records: `high_level_*_development_v39.json` and Parquet/NPZ.

---

<!-- V39_FALCON_HIGHER_ORDER_REPLICATION.md -->

# V39 — Falcon Higher Order Replication

Falcon development (80 states): second-order relative error median 0.5537 (state-bootstrap 95% CI [0.5010218089548233, 0.5826778245970267]), three-way fraction 0.5537, corrected signed projection coefficient p 0.0718, cosine 0.1399, sign stability 0.9125. Higher-order gate passes in 5/5 families; Q234 gate passes in 4/5. There are 0 below-floor states. This is a development replication, **not** V39-A qualification: validation was not opened after the mediator screen failed. Large f with small p is not signed amplification.

| Family | Falcon f | Falcon sign stability | Falcon Q234 fraction | Falcon Q234 cosine | Qwen f |
|---|---:|---:|---:|---:|---:|
| boolean_logic | 0.6038 | 1.0000 | 0.7056 | 0.8707 | 0.1784 |
| modular_arithmetic | 0.5527 | 0.9375 | 0.9123 | 0.8970 | 0.1536 |
| short_graph_traversal | 0.4132 | 0.8750 | 0.6917 | 0.8286 | 0.2218 |
| simple_state_transition | 0.4875 | 0.7500 | 0.7976 | 0.8533 | 0.1985 |
| variable_binding | 0.6257 | 1.0000 | 0.5765 | 0.7070 | 0.2063 |

---

<!-- V39_QWEN_COMPOSITION_CONTROL.md -->

# V39 — Qwen Composition Control

Qwen matched development (80 states): additive error 0.2753, second-order error 0.1965, three-way fraction 0.1965, corrected p 0.0076, sign stability 0.6625. Q234 passes 4/5; Falcon-style higher-order gate passes 0/5. This is a model comparison, not architecture-causation or a claim that Qwen is strictly additive. Validation is sealed.

---

<!-- V39_EIGHT_CONDITION_FACTORIAL.md -->

# V39 — Eight Condition Factorial

Both models completed natural R000/R100/R010/R001, then R110/R101/R011, then frozen additive and second-order predictions, then R111 on 80 fresh development states. Conv was donor-native, KV recipient-native, and exact native REC subsets were written; no later-output copy entered the factorial. Per-state six-probe vectors and predictions are in `trajectory_*_development_v39.npz` with Parquet proof records and stage freezes. Corrected `p=<I234,E111>/||E111||²`, `f=||I234||/||E111||`, and cosine obey `p=f·cos` at maximum recorded numerical error below 1e-15. Validation/final outcomes were not observed.

---

<!-- V39_INTERNAL_INTERACTION_TRACE.md -->

# V39 — Internal Interaction Trace

One frozen future probe per state was traced under all eight conditions for 20 development states/model. Falcon: 11520 state×layer×stage measurements, 26 stage/layer pairs satisfy the 0.20 ratio, 0.75 state-prevalence, 4-family and next-layer-persistence rule. Earliest persistent stages: {"ATTENTION_OUTPUT": 19, "B": 20, "C": 21, "DA": 20, "MLP_OUTPUT": 22, "POSTCONV_B_GROUP": 20, "POSTCONV_C_GROUP": 21, "POSTCONV_INPUT": 20, "POSTCONV_X": 21, "TRANSFORMED_CONTROL": 20, "TRUE_UPDATE": 20, "X": 21}. Qwen: 4480 measurements and 0 qualifying stage/layer pairs. Actual I234 and E tensors for every traced state/layer/stage are in hash-indexed NPZ banks. This is descriptive localization, not mediation.

---

<!-- V39_RESIDUAL_INTERACTION_CURVE.md -->

# V39 — Residual Interaction Curve

The residual-stream increment is computed only as `I234(block_output)-I234(block_input)` in common coordinates, never by subtracting unrelated internal spaces. Falcon median outgoing interaction norms at layers 18–23: [1.495, 2.776, 3.715, 4.405, 5.642, 11.2]. Qwen's corresponding measurements are in `residual_interaction_curve_v39.parquet` (32 decoder layers). These differences are descriptive; causal creation versus transmission is unresolved.

---

<!-- V39_INTERACTION_GENESIS_INTERVAL.md -->

# V39 — Interaction Genesis Interval

At the registered persistence threshold, Falcon attention output first qualifies at layer 19; B, dt/dA and dBx qualify at layer 20; C and x at layer 21. The common residual `BLOCK_OUTPUT` reaches 4/5-family prevalence only at the final layer 23, so it cannot satisfy the required *next-layer* persistence there. Recurrent state S and post-update S′ do not qualify by this relative threshold. A stable descriptive factor interval exists, but a unique **causal genesis interval is not identified**. First nonzero interaction and first threshold crossing are different claims. Qwen has no qualifying persistent stage on the traced subset.

---

<!-- V39_FALCON_PRIMITIVE_FACTOR_MAP.md -->

# V39 — Falcon Primitive Factor Map

Trace-selected functional sites were frozen before any primitive removal outcome. No tensor coordinate was selected post hoc.

| Family | Earliest stable layer | Eligible for pilot |
|---|---:|---|
| F1_POSTCONV_READ_FACTOR_C | 21 | True |
| F2_POSTCONV_UPDATE_FACTORS_X_B | 21 | True |
| F3_DECAY_CONTROL_DT_DA | 20 | True |
| F4_RECURRENT_STATE_INPUT_S | none | False |
| F5_RECURRENT_UPDATE_DBX | 20 | True |
| F6_OUTPUT_GATE | none | False |
| F7_COMBINED_RECURRENCE_INPUTS | none | False |
| F8_RESIDUAL_INPUT_CONTEXT | none | False |

The map is descriptive and cannot establish necessity.

---

<!-- V39_DEPENDENCY_CONSISTENT_INTERVENTIONS.md -->

# V39 — Dependency Consistent Interventions

The Falcon-native primitive hook passed 480/480 calibration state×layer no-intervention bitwise replay tests across logits, cache and five endpoint blocks. A candidate replacement occurs inside native recurrence at C, x/B, dt, S, dBx, gate or mixer input; dependent dA/dBx/S′/read/norm/mixer and all later layers are recomputed. Requested and realized replacement hashes matched in the pilot. Predictions were evaluated in float32 and cast to each native dtype; quantization gaps are recorded per intervention. Off-manifold status is explicit. Formal development mediation was **not** opened because the pilot produced no finalist.

---

<!-- V39_PRIMITIVE_FACTOR_REMOVAL.md -->

# V39 — Primitive Factor Removal

Calibration-only, first-probe screen (20 states) used natural R111 and replaced each trace-eligible native factor by its seven-condition second-order prediction at one frozen layer; all descendants recomputed.

| Candidate | Layer | Median removed direction | Median remaining norm ratio | Families passing 0.50/0.70 |
|---|---:|---:|---:|---:|
| F1_POSTCONV_READ_FACTOR_C | 21 | 0.0940 | 0.9909 | 0/5 |
| F2_POSTCONV_UPDATE_FACTORS_X_B | 21 | 0.1898 | 0.9908 | 0/5 |
| F3_DECAY_CONTROL_DT_DA | 20 | 0.0838 | 1.0106 | 0/5 |
| F5_RECURRENT_UPDATE_DBX | 20 | 0.0746 | 1.0032 | 0/5 |

All four candidates fail the preregistered pilot gate; the frozen primary finalist is `null`. No formal development removal outcome was generated. This screen does not exclude multi-layer or other mechanisms.

---

<!-- V39_STRUCTURED_FACTOR_SUFFICIENCY.md -->

# V39 — Structured Factor Sufficiency

Not executed. A structured lower-order internal baseline plus restoration of only one candidate factor's natural three-way component requires a frozen candidate. The calibration screen selected none. Adding the exact endpoint interaction back would be tautological and is not counted as sufficiency. Validation and final remain sealed.

---

<!-- V39_Q4_ROTATION_DECOMPOSITION.md -->

# V39 — Q4 Rotation Decomposition

Development median cosine between isolated-Q4 and Q4-given-Q23 future effects: Falcon 0.6809; Qwen 0.9641. Falcon's median context-conditioned Q4 magnitude ratio is 0.9969. Native internal x/B/C/dt/attention/MLP interaction tensors are recorded, but no component-specific prospective causal rotation prediction was frozen. This is a descriptive rotation, not an explanation.

---

<!-- V39_Q4_ROTATION_CAUSAL_TEST.md -->

# V39 — Q4 Rotation Causal Test

Not executed: no primitive finalist passed the calibration screen, and no separate Q4 causal component/prediction was frozen. The required local cosine≥0.90, downstream rotation≥50%, 4/5 families, and development+validation gate is therefore untested, not failed or passed.

---

<!-- V39_ATTENTION_MLP_ALTERNATIVES.md -->

# V39 — Attention Mlp Alternatives

Falcon attention output reaches the internal 0.20/0.75/4-family persistence threshold at layer 19, before several traced post-Conv factors at layers 20–21. MLP output qualifies at layer 22. These are credible alternative carriers, not identified causal generators. No attention/MLP-specific removal or structured restoration was run, so `ATTENTION_MEDIATED`, `MLP_MEDIATED`, and `RESIDUAL_NONLINEAR_COMPOSITION` remain open hypotheses. Qwen has no qualifying persistent internal stage on this subset.

---

<!-- V39_MIXER_RESIDUAL_CEILINGS.md -->

# V39 — Mixer Residual Ceilings

On 20 development trace states (one probe), each Q4 layer 18–23 was tested separately by subtracting that boundary's measured I234 from the natural R111 mixer or block output, then allowing downstream native computation. Across all tested layers, mixer-output median removed direction 0.1298, remaining norm ratio 0.9717; block-output values 0.2235 and 0.9129. These are single-layer **interface ceilings**, not primitive mediation or a joint multi-layer test. Per-layer records are in `interface_ceilings_F_development_v39.parquet`.

---

<!-- V39_MODEL_COMPARATIVE_PROFILE.md -->

# V39 — Model Comparative Profile

Matched development shows Falcon f=0.5537, second-order error=0.5537, Q4 context cosine=0.6809; Qwen f=0.1965, error=0.1965, cosine=0.9641. Both high-level REC gates and both Q234 gates pass, but only Falcon passes the frozen higher-order gate on development. Matched design does not identify architecture as the cause. Validation was not opened, so V39-I is development-supported only.

---

<!-- V39_INDEPENDENT_FINAL_OPENING.md -->

# V39 — Independent Final Opening

The independently generated Falcon final pool of 40 states was sealed before outcomes and remains unopened. The frozen rule requires development+validation high-level/Q234/higher-order replication **and** one frozen primitive mediator passing necessity and structured sufficiency in both roles. No finalist qualified the calibration screen, no formal mediation was run, and validation stayed sealed. Qwen final also remains sealed. No final-intervention outcome exists.

---

<!-- V39_TASK_GROUNDED_DESIGN.md -->

# V39 — Task Grounded Design

Task-grounded Phase C did not open because the prerequisite internal mediator did not qualify. The proposed Type S surface-only, Type T task-variable, and Type N neutral-natural-token contrasts were not instantiated or tested. No model-output-dependent task labels were created. A future study must preseal externally generated truth values before model execution.

---

<!-- V39_TASK_GROUNDED_RESULTS.md -->

# V39 — Task Grounded Results

Not tested. No Type S/T/N task-grounded panel was opened, and no correct-answer-margin or task-variable classification outcome was measured. V39-J is unsupported/untested; donor-response similarity is not task correctness.

---

<!-- V39_SEMANTIC_SURFACE_CONTROLS.md -->

# V39 — Semantic Surface Controls

Not executed because task-grounded Phase C was sealed. There are no Type S surface-only, Type T task-variable, or Type N neutral-token contrast outcomes. The development control panel does include same-prefix wrong-token and off-manifold shuffled REC controls, which are **not** substitutes for semantic Type S/T/N controls.

---

<!-- V39_STRICT_INTERFACE_AUDIT.md -->

# V39 — Strict Interface Audit

V39 calibration: Falcon 20 and Qwen 20 states each passed native/instrumented bitwise replay, exact REC writeback and actual eight-condition formula equality. Maximum algebra residuals: F 6.66e-16, Q 1.78e-15. Expanded trace hook passed 20/20 Falcon and 20/20 Qwen bitwise tests; Falcon primitive hook passed 480/480 state×layer tests. Six-probe endpoint predictions were frozen before R111; recipient KV and donor Conv were unchanged by REC subset interventions. Off-manifold shuffling/sign-flip and interface-copy controls are explicitly non-natural. This is a V39-specific audit, not a reuse of V36 results.

---

<!-- V39_EXECUTION_MANIFEST.md -->

# V39 — Execution Manifest

The machine record `results/v39/processed/v39_execution_manifest.json` enumerates stage outcomes and freeze artifacts. Stages 0–6 completed, with a frozen null finalist; formal mediator development, validation, independent final and Phase C were not opened. All 180 fresh pool programs/prompts are mutually disjoint and historical-disjoint; formal horizon is constant across roles. `v39_integrity_index.json` hashes code, configs, pools, stage seals, machine results, reports, and retained local NPZ banks. NPZ tensor banks are intentionally not committed to Git, but their hashes and paths are committed.

---

<!-- V39_SCIENTIFIC_ANSWERS.md -->

# V39 — Scientific Answers

1. V38 projection label was dimensionally wrong; f and sign were sound.

2. Yes; append-only V38 metric amendment added.

3. Falcon high-level effect passes development; validation unobserved.

4. Natural Q234 passes 4/5 families in Falcon development.

5. Falcon higher-order gate passes 5/5 families in development.

6. Validation not opened after null mediator finalist.

7. Independent final remains sealed.

8. Yes on development: Qwen f 0.197 versus Falcon 0.554.

9. Attention output first persistently crosses registered internal ratio threshold at layer 19.

10. No registered persistent substantial Q2 interval identified.

11. No registered persistent substantial Q3 interval identified.

12. Several Falcon Q4 factors qualify at layers 20–22.

13. Incoming residual is not the earliest registered persistent stage; causal precedence unresolved.

14. Post-Conv B qualifies at layer 20, C and x at 21, descriptively.

15. dt/dA and dBx qualify at layer 20, descriptively.

16. S′ does not pass registered next-layer persistence.

17. Raw read does not pass registered persistence.

18. Gating/norm carries interaction, but no causal generator identified.

19. Residual BLOCK_OUTPUT reaches four-family prevalence only at final layer, without next-layer persistence.

20. Attention output qualifies at layer 19, descriptive only.

21. MLP output qualifies at layer 22, descriptive only.

22. Attention is earliest qualifying traced stage; among primitive factors B/dt/dBx appear by 20.

23. Tested single-layer calibration clamps do not reach the 0.50 directional gate.

24. Best screened median removed direction is 0.190 (x+B); not qualifying.

25. Norm and direction diverge; no strong removal mechanism identified.

26. Structured restoration was not opened.

27. Necessity/sufficiency classification untested formally.

28. Joint multi-layer factor requirement remains possible, untested.

29. C/read mediation pilot fails; Q4 causal decomposition untested.

30. x+B pilot fails the registered threshold.

31. dt/dA pilot fails the registered threshold.

32. Residual input context did not qualify trace-site screen; not causally tested.

33. No prospective native Q4 component rotation prediction frozen.

34. No Q4 component-specific causal rotation test executed.

35. Trace shows recurrence, attention and MLP carriers; creation not causally assigned.

36. Qwen has no registered persistent internal stage on the traced subset.

37. Qwen high-level channel phenomenon and Q234 pass development; mechanism unknown.

38. V39-A not qualified: validation sealed.

39. V39-B not tested: final sealed.

40. V39-C descriptive localization only, no causal interval.

41. V39-D/E/F/G unconfirmed; primitive pilot fails and alternative paths untested.

42. V39-H untested causally.

43. V39-I development comparison supported, validation not performed.

44. Task-grounded Phase C did not open.

45. Type T untested.

46. Type S untested.

47. V39-J untested.

48. Mechanism unresolved on development; full V39-K requires stronger replication.

49. No precise mechanism is ready for a training-origin study.

50. A future temporal Conv→REC study is not yet justified by V39 mechanism evidence.

These answers distinguish development evidence from unobserved validation/final outcomes; no skipped phase is called a negative experiment.

---

<!-- V39_COMPLETE_REPORT.md -->

# V39 — Complete Report

V39 reached its prospectively specified stop at Stage 6: a null primitive finalist after calibration-only screening. It did **not** complete the conditional validation, independent-final, or task-grounding branches. Falcon development reproduces the high-level REC effect, Q234 effect, and higher-order three-way endpoint response (f=0.5537, corrected p=0.0718; 5/5 higher-order families). Qwen development has a lower three-way fraction (f=0.1965). Internal tracing locates late Falcon factor and attention interactions, but causal origin remains unresolved. Four trace-selected single-layer native-factor screens fail the 0.50/0.70 pilot gate; no formal mediator or structured sufficiency result exists. The independent final remains sealed. The strongest defensible conclusion is **development interaction replicated; mechanism unresolved; full V39-A/B/K not independently confirmed**.

Limitations: one-probe trace/pilot, four calibration states per family, single-layer factor screening, no joint multi-layer mediation, no prospective attention/MLP mechanism test, and no validation/final. These omissions are explicit consequences of the frozen gate, not hidden negative findings. V38 history is immutable; the projection amendment is append-only. See `v39_adjudication.json` and the integrity index for machine audit.
