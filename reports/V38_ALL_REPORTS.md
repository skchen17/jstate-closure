# V38 — All Reports

# V38 — Frozen Starting Point

Parent commit `ef075e85025eb634b0a02fecadc16c8a4a6811e9`; V1–V37 reports/results unchanged and V37 outcomes excluded from V38 formal decisions. New pools contain calibration/development/validation/final = 20/80/40/40 states per model, five families, all cross-role program/prompt-disjoint and historically disjoint. Validation/final were sealed before intervention responses. Pool hashes and model/checkpoint/tokenizer hashes are in `artifacts/trajectory_composition_v38.freeze.json` and `data/v38/sample_pool_manifest_v38.json`.

Important distribution shift: modular arithmetic uses horizon 2 in calibration/development and horizon 3 in validation/final, after a sealed, outcome-blind amendment. This limits same-distribution interpretation. V18 training leftovers were not used in formal roles.


---

# V38 — High Level Reconfirmation

Fresh donor Conv + donor REC versus donor Conv with recipient REC, with recipient-native KV and six-probe endpoints:

|model|role|n|positive|median reduction|families|
|---|---|---|---|---|---|
|Q|development|80|0.988|0.435|5|
|Q|validation|40|0.975|0.392|5|
|F|development|80|1.000|0.383|5|
|F|validation|40|1.000|0.457|5|

All four model×role cells pass ≥0.80 positive, ≥0.20 median reduction and ≥4 families. REC-only weakness on the frozen control subset is separately shown in `V38_STRICT_INTERFACE_AUDIT.md`.


---

# V38 — Q234 Natural Replication

R111 is exact native Q2+Q3+Q4 donor REC writeback on donor Conv; no future outputs copied. Recovery is donor-error gain relative to the full REC+Conv gain (states with benefit >1).

|model|role|Q234/full benefit|cos to full|families ≥ thresholds|
|---|---|---|---|---|
|Q|development|0.762|0.904|5|
|Q|validation|0.814|0.902|5|
|F|development|0.783|0.831|3|
|F|validation|0.833|0.895|4|

Aggregate medians are strong in both models/roles. The additionally frozen 4/5 family Q234 gate fails for F development (3/5), so broad family-uniform replication must not be claimed for that cell.


---

# V38 — Factorial Design

Fixed background: donor Conv, recipient REC, recipient-native KV. Q2/Q3/Q4 are the 7–12/13–18/19–24 recurrent-layer groups. Conditions: `R000`, `R100`, `R010`, `R001`, `R110`, `R101`, `R011`, `R111`; group membership and six probes are in `execution_plan_v38.json` and model designs. All algebra uses state-wise six-probe vectors.

`Eabc=Yabc−Y000`; additive=`E100+E010+E001`; second-order=`E110+E101+E011−E100−E010−E001`; three-way=`E111−E110−E101−E011+E100+E010+E001`. The pasted expressions contained typographic `*` artifacts; the standard inclusion–exclusion formulas were frozen before outcomes.


---

# V38 — Eight Condition Rec Factorial

Every development (80/model) and validation (40/model) state has all eight REC conditions. Before R111, singles and pairs were sealed and both predictions hashed. Each condition uses exact native REC-state replacement only; Conv/KV were checked unchanged, and all later computations evolved natively. State-wise outputs are in `trajectory_*_v38.parquet` and local ignored `.npz` arrays; per-condition REC hashes are in the `proof_json` column.


---

# V38 — Additive Prediction

Prospective additive prediction of R111 from unopened singles:

|model|role|add err|add cos|2nd err|2nd cos|improve|family passes|
|---|---|---|---|---|---|---|---|
|Q|development|0.263|0.971|0.161|0.988|0.330|{'additive': 1, 'higher_order': 0, 'pairwise': 3}|
|Q|validation|0.253|0.972|0.184|0.983|0.297|{'additive': 1, 'higher_order': 0, 'pairwise': 2}|
|F|development|0.656|0.864|0.602|0.833|0.059|{'additive': 0, 'higher_order': 5, 'pairwise': 0}|
|F|validation|0.531|0.885|0.496|0.881|0.055|{'additive': 0, 'higher_order': 5, 'pairwise': 0}|

Q has low median additive error but material second-order improvement, so it does not pass the registered additive class. F additive error/cosine fail directly. Bootstrap CIs are stored per model/role in `trajectory_analysis_*_v38.json`.


---

# V38 — Pairwise Interactions

Exact state-wise pair terms and donor-effect-direction projections (negative projection is cancelling, not amplifying):

|model|role|||I23||/||E|||||I24||/||E|||||I34||/||E|||proj I23|proj I24|proj I34|
|---|---|---|---|---|---|---|---|
|Q|development|0.189|0.122|0.150|-0.010|-0.052|-0.084|
|Q|validation|0.189|0.141|0.156|-0.098|-0.100|-0.144|
|F|development|0.471|0.428|0.442|-0.455|-0.436|-0.428|
|F|validation|0.394|0.370|0.376|-0.282|-0.397|-0.302|

F shows large, mostly cancelling pair contributions; Q pair terms are smaller. The largest median norm term is I23 in both models, but no serial-circuit interpretation follows. Full distributions and per-family metrics are machine-readable.


---

# V38 — Second Order Prediction

Predictions were sealed before R111.

|model|role|add err|add cos|2nd err|2nd cos|improve|family passes|
|---|---|---|---|---|---|---|---|
|Q|development|0.263|0.971|0.161|0.988|0.330|{'additive': 1, 'higher_order': 0, 'pairwise': 3}|
|Q|validation|0.253|0.972|0.184|0.983|0.297|{'additive': 1, 'higher_order': 0, 'pairwise': 2}|
|F|development|0.656|0.864|0.602|0.833|0.059|{'additive': 0, 'higher_order': 5, 'pairwise': 0}|
|F|validation|0.531|0.885|0.496|0.881|0.055|{'additive': 0, 'higher_order': 5, 'pairwise': 0}|

Q second-order errors are low, but the ≥30% improvement and ≥4-family pairwise gate is not met (family counts 3/5 development, 2/5 validation). F remains well above 0.30 relative error. Thus no model satisfies the formal pairwise class in both roles.


---

# V38 — Three Way Interaction

|model|role|median fraction|median projection|sign stability|higher family pass|
|---|---|---|---|---|---|
|Q|development|0.161|0.048|0.600|0|
|Q|validation|0.184|0.096|0.650|0|
|F|development|0.602|0.822|0.963|5|
|F|validation|0.496|0.619|0.925|5|

F has a large, positive, stable residual beyond second order in both roles; Q does not. This is an interaction identity, not a localized circuit proof.


---

# V38 — Composition Class Adjudication

Development: Q = no registered class, F = higher-order (5/5 families). Validation: same pattern. No single class passes development+validation in both models. Therefore cross-model A/B/C/E/F are not confirmed; the preregistered independent final is not eligible and remains sealed (40 states/model). `final_gate_v38.json` records this decision. At cross-model scope the outcome is no shared stable rule; F-specific higher-order replication is a secondary architecture-specific observation, not a shared formal class.


---

# V38 — Incremental Natural Accumulation

The cumulative sequence is R000→R100→R110→R111. Per-state increments are Δ2=`E100`, Δ3|2=`E110−E100`, Δ4|23=`E111−E110`, compared against isolated `E010` and `E001`. These are re-expressions of the frozen eight-condition vectors, not extra experiments. Median isolated regional effect norms:

|model|role|Q2|Q3|Q4|Q23|Q234|
|---|---|---|---|---|---|---|
|Q|development|4.748|6.507|6.669|8.962|11.440|
|Q|validation|4.090|5.863|6.067|7.642|9.969|
|F|development|3.987|5.997|4.156|6.646|7.491|
|F|validation|4.283|6.900|4.441|8.267|8.958|


---

# V38 — Context Dependent Marginal Effects

Q3 given Q2 versus isolated Q3, and Q4 given Q2+Q3 versus isolated Q4:

|model|role|cos Q3 alone|Q2|cos Q4 alone|Q23|Q3 magnitude ratio|Q4 magnitude ratio|
|---|---|---|---|---|---|
|Q|development|0.947|0.961|0.998|0.995|
|Q|validation|0.952|0.968|0.987|0.996|
|F|development|0.800|0.621|0.982|1.009|
|F|validation|0.858|0.686|1.006|1.008|

F shows stronger directional context dependence, especially for Q4; ratios near one show that rotation can dominate without large magnitude change. Additional Q4|Q2 and Q4|Q3 cosine columns are in the analysis Parquet files. No seriality is inferred.


---

# V38 — Marginal Effect Rotation

Same-state cosine and magnitude ratios for conditional versus isolated regional increments:

|model|role|cos Q3 alone|Q2|cos Q4 alone|Q23|Q3 magnitude ratio|Q4 magnitude ratio|
|---|---|---|---|---|---|
|Q|development|0.947|0.961|0.998|0.995|
|Q|validation|0.952|0.968|0.987|0.996|
|F|development|0.800|0.621|0.982|1.009|
|F|validation|0.858|0.686|1.006|1.008|

Q directions remain near aligned (median cosines ≈0.95–0.97); F Q4|Q23 rotates more (≈0.62 development, 0.69 validation), while median magnitude ratios remain near 1. This is a vector-space property of the tested interventions only.


---

# V38 — Cross Model Composition

Q: additive prediction is fairly accurate but second-order gains are material and the pairwise 4/5-family rule fails; no stable registered class. F: second-order prediction remains poor, while a positive three-way term is stable in 5/5 families across development and validation. Numeric equality was never required. The models do not share a winning qualitative class; a universal correction/composition mechanism is not established.


---

# V38 — Q1 Control

Frozen 4/family development and 2/family validation control subsets compare natural Q234 versus Q1+Q234 (all donor REC on donor Conv):

|model|role|n|Q234 donor error|all Q donor error|paired Q1 error gain|Q1 vector increment|
|---|---|---|---|---|---|---|
|Q|development|20|11.202|9.165|1.393|5.871|
|Q|validation|10|8.943|7.733|1.495|5.607|
|F|development|20|8.567|7.707|0.454|3.906|
|F|validation|10|7.275|6.645|0.710|3.236|

Q1 is not dispensable on these subsets: adding it further reduces donor error. This is secondary and does not reopen layer search.


---

# V38 — Trajectory Internal Trace

Five frozen development states/model (one/family), all eight conditions, first frozen probe: 40 hash-only rows/model. Q2/Q3/Q4 boundary hidden, Conv/REC incoming states, post-Conv factors, transformed controls, true update, recurrent read, mixer output, and logits were captured as hashes in `trajectory_trace_*_development_v38.parquet`. The Q/F boundary layer indices are in trace summaries. These are explanatory one-probe traces, not the formal six-probe composition evidence; no future output was copied.


---

# V38 — Dose Response

F-only 20-state diagnostic after its development higher-order class was frozen. λ=0.25/0.5 interpolations are off natural trajectories; λ=1 exactly replays natural R111 (max absolute discrepancy 0.000).

|λ|additive relative error|second-order relative error|
|---|---|---|
|0.25|1.291|1.333|
|0.5|0.908|0.902|
|1.0|0.614|0.603|

The small-λ conditions are not more additive here; error is larger than at λ=1. This does not support the proposed small-perturbation-additive → natural-nonlinear pattern. No Q dose class was opened because Q had no frozen development class. V38-D is not confirmed.


---

# V38 — Jvp Finite Comparison

Optional JVP was not run. The F interpolation diagnostic cannot substitute for a local Jacobian. No claim about local linearization failure is made.


---

# V38 — Task Grounded Design

Phase B was authorized because F passed a development composition class and the high-level gate held in both models. `task_design_v38.json` froze 4/family development and 2/family validation states, generator-derived first answer action, model-specific one-token candidates, and one common newline probe before task-specific outcomes. All eight REC conditions plus recipient/REC-only/joint/donor/KV-only were measured. Crucial limitation: no controlled Type S or Type T semantic/surface prompt forks; this design cannot establish strict task-variable mediation.


---

# V38 — Task Grounded Results

One-token answer diagnostic (invalid/coverage-fail states excluded):

|model|role|valid|margin positive|median margin Δ|R000 accuracy|R111 accuracy|
|---|---|---|---|---|---|---|
|Q|development|18|0.389|-0.125|0.833|0.778|
|Q|validation|9|0.111|-0.250|0.778|0.778|
|F|development|18|0.444|-0.016|0.111|0.111|
|F|validation|9|0.333|-0.016|0.222|0.222|

R111 does not consistently improve the externally defined correct-answer margin or accuracy over R000. Neither model shows a stable task-grounded effect. Probe/format and missing semantic contrasts further limit interpretation; V38-G is not established.


---

# V38 — Semantic Surface Controls

No valid Type S surface-only or Type T task-variable-changing same-prefix forks were created in V38. The wrong-token same-prefix REC control was run in `controls_*_v38.parquet`, but it is not a substitute for a controlled task-variable contrast. Accordingly, donor-like vectors are not interpreted as semantic content.


---

# V38 — Channel Task Function

The Task Phase B diagnostic includes recipient, REC-only, Conv-only R000, REC+Conv joint, KV-only, and Q234 with recipient-native KV. Per-condition one-token answer accuracies are in `task_diagnostic_*_v38.json`. With no positive task-grounded mediation and no Type S/T forks, it does not identify a functional channel specialization. REC-only is weak in donor-vector error on frozen control subsets, a distinct measurement.


---

# V38 — Temporal Delay Design

Phase C required stable task-grounded Phase B evidence; this gate failed. No delay sequence was selected or executed. Model Conv-window size and suffix-flush behavior were not inferred.


---

# V38 — Temporal Channel Profile

Not opened: no time-dependent Conv/REC/KV causal profile was measured. V38-H and any claim of information transfer to REC are unsupported.


---

# V38 — Strict Interface Audit

Exact native REC writeback, donor Conv preservation, recipient-native KV, instrumented bitwise replay and six-probe continuation passed fresh V38 calibration. On frozen control subsets, median donor errors:

|model|role|recipient|REC-only|Conv-only|joint|wrong token|shuffle*|signflip*|read ceiling|
|---|---|---|---|---|---|---|---|---|---|
|Q|development|50.413|50.347|16.343|9.165|14.978|111.079|24.390|10.744|
|Q|validation|53.036|48.865|15.917|7.733|14.929|107.942|26.292|8.598|
|F|development|52.228|51.244|13.404|7.707|14.235|85.482|21.117|8.056|
|F|validation|52.151|52.406|12.391|6.645|12.587|86.923|22.644|6.833|

`*` Off-manifold controls, not natural states. Wrong token is same-prefix natural alternative. Q234 read-interface output patch is explicitly an interface ceiling and never counted as a natural mechanism. Raw tensor arrays remain local/ignored by Git; table records, hashes and seals are versioned.


---

# V38 — Execution Manifest

Base freeze `0a257bad114ebb3364d8fb1e407b94df703a968e854e68f7cd984ea8de0d5b83`; design freeze `3fa6926cc59a330c2d67b85882a40be30b075afd91d84215948b71cf6cb5e79b`; development adjudication freeze `795d5f94cd9cd180172202e82408ad17bf712d696a20ea0f17d37be86aa67237`; validation adjudication freeze `424ca77e4c0abc5ce7c022f1ed69b8878cf5297cf4221ed33baab527a3d3beb8`; final-closed freeze `b542660dfe7beb0d9b427cd0316b52af01222fe2541374f45f15e8c0e2037f30`.

For every model×role, single conditions preceded pair conditions, both predictions were sealed before R111, and stage timestamps were verified. Independent final was never executed. Machine paths/hashes are listed in `v38_integrity_index.json`; `.npz` vectors are retained locally and ignored by Git per repository policy.


---

# V38 — Scientific Answers

1–2. High-level conditional benefit and aggregate natural Q234 restoration replicate in both models; F development fails the additionally frozen 4/5 Q234 family-uniform gate.

3–8. Isolated Q2/Q3/Q4 and pair responses were measured as same-state vectors; exact norms/distributions are in analysis Parquet files.

9–12. Q additive prediction is close but not a formal additive-class pass; F additive prediction fails. Exact median errors/cosines are in the additive report.

13–17. I23/I24/I34 are explicit vectors. F pair terms are large and generally cancelling; I23 has the largest median norm in both models. No serial-circuit inference.

18–22. Q second-order fit is good but family/improvement gate fails; F second-order fit fails. F three-way residual is large, positive and sign-stable; Q three-way is not stable.

23–24. Q and F do not require the same registered composition class; no shared cross-model class passes.

25–29. Conditional Q3/Q4 increments differ from isolated increments, mainly by stronger directional rotation in F; ratios are near one.

30. Q1 adds a measurable donor-directed increment on the frozen control subsets.

31–33. F-only off-manifold dose interpolation does not become more additive at small λ; V38-D is unsupported. JVP was optional and not run.

34–37. Donor-vector fidelity is strong, but the generator-defined correct-answer diagnostic does not improve consistently; no semantic/surface Type S/T contrast or task-grounded channel function is established.

38–40. Temporal phase remained closed; Conv-window or REC persistence claims were not tested.

41–48. Cross-model V38-A/B/C/E/F/G/H not confirmed; V38-D unsupported. At cross-model scope V38-I (no shared stable rule) applies, while F has a replicating within-model higher-order pattern.

49. No cross-model composition rule is licensed for a third-model frozen test. F-specific three-way sign/fraction is a candidate only, not a shared law.

50. A targeted exploratory F training-origin study is scientifically permissible after its replicated model-specific pattern; a shared-law V39 training-origin study is not yet justified.


---

# V38 — Trajectory-Level Causal Composition of Persistent State

## Result

Both models replicate the high-level REC–Conv effect and broad natural Q234 reconstruction. F has a prospectively replicated, positive three-way Q2/Q3/Q4 interaction (5/5 families development and validation). Q has no registered additive/pairwise/higher-order class despite relatively accurate second-order vector prediction. No shared cross-model class qualifies, so independent final remains sealed (40 states/model).

## Prediction evidence

|model|role|add err|add cos|2nd err|2nd cos|improve|family passes|
|---|---|---|---|---|---|---|---|
|Q|development|0.263|0.971|0.161|0.988|0.330|{'additive': 1, 'higher_order': 0, 'pairwise': 3}|
|Q|validation|0.253|0.972|0.184|0.983|0.297|{'additive': 1, 'higher_order': 0, 'pairwise': 2}|
|F|development|0.656|0.864|0.602|0.833|0.059|{'additive': 0, 'higher_order': 5, 'pairwise': 0}|
|F|validation|0.531|0.885|0.496|0.881|0.055|{'additive': 0, 'higher_order': 5, 'pairwise': 0}|

Q234 natural reconstruction:

|model|role|Q234/full benefit|cos to full|families ≥ thresholds|
|---|---|---|---|---|
|Q|development|0.762|0.904|5|
|Q|validation|0.814|0.902|5|
|F|development|0.783|0.831|3|
|F|validation|0.833|0.895|4|

F development does not meet the extra 4/5-family Q234 reconstruction gate (3/5); the modular-arithmetic horizon shifts from 2 to 3 in validation/final. Both limitations constrain inference.

## Secondary phases

The frozen controls confirm exact native intervention semantics; Q1 is not negligible. F-only off-manifold interpolation does not support a small-λ additive regime. Task answer-token diagnostics do not show stable improvement and lack Type S/T contrasts; task-grounded V38-G is not established. Temporal Phase C and optional JVP were not run.

## Audit

All roles were generated afresh, historically disjoint and sealed before formal outcomes. For each model/role, singles→pairs→hashed additive/second-order predictions→R111 ordering was verified. V1–V37 were not edited. Machine outputs and hashes are in `results/v38/processed/v38_integrity_index.json`. See `V38_ALL_REPORTS.md` for every topic in one file.
