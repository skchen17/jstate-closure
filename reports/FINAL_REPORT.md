# J-State Closure Final Report

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Mode: `cumulative evidence adjudication through v6`
- Date: `2026-09-15`
- Verification Status: `repository-grounded; machine results cited below`
- Protocol: `peripheral_foundations_protocol_v6`
- causal/null freeze digest: `f62c55d21e5e220cd21ea81c19a28bf1d5249a60bdc35a08e3eb029dd61ec12e`
- strong ceiling freeze digest: `6d4c77a3e33edf9e977e04293b73b77d153a15af091b1c496fe869e92ec23170`
- compact freeze digest: `07b51492b188b23da2ad66708cd3520e7b568a65bfd867ec019da7d301c24417`
- delivery/report freeze digest: `160d746a5b9b14e226acf876989e3196f971f321dbb9d2fb3720f7da68319840`


## Current adjudication

The strongest warranted conclusion remains **H2 for the tested operational measured-J state**, with evidence that the pooled and Boolean effects are predominantly mediated by later measured-J writes under the tested restoration operator. The full operational remainder contains learnable causal-direction information, but no compact peripheral state passed conditional sufficiency. This does **not** establish H3, an autonomous controller, consciousness, “true thoughts,” or parameter localization.

## Required v6 questions

1. **Is the repaired teacher delta stable?** Yes: non-zero float32 fraction `1.000`, median f32/f64-projection cosine `1.000000`.
2. **How much of v5 failure was float16?** The old float16 endpoint had 2/66 nonzero vectors, while the float32 re-audit finds only 2/66 material effects at that same next-token/layer-23 endpoint. Therefore most zeros reflect the endpoint's causal structure, not float16 alone. Float16 nevertheless perturbed stored intervention candidates by median L2 `0.035538`, which likely explains the v5 mediation inconsistency.
3. **Does restoration alone cause an effect?** No detectable clean-state artifact: pooled JS `0.000000 [-0.000000, 0.000000]`.
4. **Corrected v4/v5 mediation?** Pooled single JS `0.002166 [0.000181, 0.005774]` falls to corrected persistent `0.000167 [0.000019, 0.000435]` (M=`0.922975`). Boolean falls from `0.005749 [0.000325, 0.015569]` to `0.000050 [0.000019, 0.000089]` (M=`0.991279`).
5. **H2-A, current J insufficiency?** Yes for the tested state: pooled and Boolean single-arm effects clear the frozen numerical floor; Boolean is the strongest family.
6. **H2-B, mediation by later J writes?** Supported for pooled/Boolean under this operator (92.3%/99.1% point-estimate removal), but not universally: state-transition has only six items, fails the single-effect noise gate, and is amplified by restoration. State-dependent restoration distortion remains a limitation not measured by a clean-state null.
7. **Does full remainder contain learnable information?** Yes for causal directions/projections: causal-direction gain `0.456251 [0.379585, 0.538002]` and ordinary causal-projection RMSE improvement `0.002717 [0.002231, 0.003196]`. Semantic/output benefits are absent or uncertain.
8. **Most informative endpoints?** Intervention-sensitive J directions, causal-projection RMSE, and next-J cosine. Semantic accuracy decreases and output-sign gain crosses zero.
9. **Which families?** Causal-direction gains are positive in all five families; short graph is largest. Global next-J gains are largest in modular arithmetic/state transition and absent in variable binding. Sample sizes for causal family estimates are only 3–12.
10. **Strong ceiling?** `True`.
11. **Smallest effective C?** No validated sufficient C. The smallest/only joint screen-pass is `pca 512D` if “effective” means predictive+causal gap screening only.
12. **Causal fidelity and conditional sufficiency?** Screen causal fidelity reaches cosine `0.468382` and closes `0.997032` of the full causal gap, but conditional residual adds `0.009032` causal cosine and `2.26` semantic percentage points. It therefore fails sufficiency.
13. **Ready for recurrent controllers?** No. v6 intentionally trained none; entry additionally requires a strong ceiling, a compact gap-closing C, stable causal fidelity, and conditional sufficiency.

## Evidence classes

- **Causal:** paired teacher interventions and persistent/null arms.
- **Predictive/associational:** held-out one-step peripheral-reference gains.
- **Numerical validation:** float32 storage and float64 projection sensitivity.
- **Practical magnitude:** raw JS/J/action effects and CIs, not p-values alone.

## Validity/fallacy scan

1. Correlation/causation: predictive ceiling is not called causal.
2. Measurement validity: finite 4096D measured-J is named operationally.
3. Aggregation: pooled and family-wise effects are separate.
4. Selection leakage: causal directions use fit; strong architecture selection uses ordinary validation only; fidelity uses causal test. The compact sweep is exploratory and its 18-way test-set screen is not presented as confirmatory.
5. Optional stopping: frozen gates determine compact authorization.
6. Null-result interpretation: weak models do not prove absent information.
7. Scale conflation: small global energy and causal importance are separate.
8. Ratio instability: mediation ratios are gated on non-noise single effects.
9. Intervention naturality: inherited validated v5 candidate criteria remain explicit.
10. Restoration confounding: clean-state null limitation is stated.
11. Overclaiming: no consciousness, true-thought, parameter-localization, or autonomous-controller claim.

## Next blocker

The strong ceiling is now established for causal-sensitive endpoints. The blocker is a compact `C_t` that retains these effects under an independent confirmation split and passes predictive, causal, and semantic conditional-sufficiency tests with CIs. Recurrent training remains blocked until then.

## Exact v6 commands

```bash
scripts/run_peripheral_v6.sh freeze --run-suffix protocol-freeze-r3
CUDA_VISIBLE_DEVICES=0 scripts/run_peripheral_v6.sh causal --run-suffix full-66
scripts/run_peripheral_v6.sh merge --run-suffix merge-full66
scripts/run_peripheral_v6.sh freeze --run-suffix leakfree-ceiling-freeze-r6
CUDA_VISIBLE_DEVICES=1 scripts/run_peripheral_v6.sh ceiling --run-suffix leakfree-strong
scripts/run_peripheral_v6.sh freeze --run-suffix matched-compact-freeze-r7
CUDA_VISIBLE_DEVICES=1 scripts/run_peripheral_v6.sh compact --run-suffix compact-sweep-matched
scripts/run_peripheral_v6.sh freeze --run-suffix delivery-freeze-r12
scripts/build_report_v6.sh --run-suffix final-reports-r5
```
