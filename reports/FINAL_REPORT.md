# Final report

## Material Passport

- Protocol: `phase0_protocol_v2` (frozen before fresh confirmation)
- Phase 0 v2 gate: **PASSED**
- Closure-layer calibration gate: **FAILED**
- Formal downstream phases: **GATED / NOT EXECUTED**
- Strongest warranted conclusion: **D — measurement quality is insufficient to distinguish H1, H2, and H3**

This report makes no claim about consciousness or extraction of a model's “true
thoughts.” “Measured-J component” and “measured-J remainder” refer only to the
declared finite dictionaries and sparse decomposition.

## Verified measurement results

Fresh official-compatible pass@10 was 0.775862 for multihop (58 items) and 0.716797 for order of operations (256 items). The item-clustered MRR advantage was 0.135202, 95% CI [0.109490, 0.161532]. At frozen layer 24, the intended-answer log-odds effect was 3.064270, 95% CI [1.978771, 4.221590], above the 0.0001 null envelope.

Independent closure-layer calibration then tested layers 23, 24, 25, 26, 27, 28, 29. It obtained 0/1400 strictly valid clamp trials; the best layer-level valid rate was 0.000, below the frozen 0.80 requirement. All candidate layers therefore failed only the clamp-valid-rate criterion, and the eligible set is empty.

The v2 Phase 0 result is statistically positive and practically above its frozen readout
thresholds. It does not supersede the independent causal-state gate. Strict-all-layers
hit@10 remains a sensitivity analysis, not the v2 primary statistic.

## Answers to the 15 adjudication questions

1. **Did the fresh Phase 0 readout gate pass?** Yes. The exact pass@10 and confidence intervals are reported above.
2. **Was a lens-quality band identified?** Yes: block-output layers 20–30, selected from calibration only.
3. **Did any layer become closure eligible?** No. Layers 23–29 passed readout, rank, positive-control, numerical, and repeatability checks, but each had 0/200 strictly valid clamps.
4. **Is instantaneous measured-J approximately Markov sufficient?** Undetermined; no gate-authorized closure trial was run.
5. **Does measured-J remainder causally influence future measured-J?** Undetermined; calibration failure prevents a causal estimate.
6. **Is any influence mediated by later measured-J writes?** Undetermined; one-shot, final-persistent, and all-position-persistent mediation arms were gated.
7. **Did final-token and sequence-state arms agree?** Not tested. Their scope hooks are implemented and tested, but neither arm produced empirical effects.
8. **Does E_R decrease as dictionary size grows from 4,096 to 16,384?** Not tested. Nested dictionaries were built, but no common-valid paired Phase 3 trials exist.
9. **Do natural collisions reproduce a remainder association?** Not tested; no observational collision bank was built after the layer gate failed.
10. **Can short layer-depth J history close the oracle gap?** Not tested.
11. **Can token-time J plus compact recurrent memory close the gap?** Not tested.
12. **What is the smallest stable autonomous controller?** None established; controller training was gated.
13. **Does a controller generalize to unseen procedural tasks?** Not tested.
14. **Does external knowledge restore knowledge-heavy performance?** Not tested.
15. **Was teacher/student latent-intervention fidelity demonstrated?** No; Phase 6B was not executed.

## Evidence by type

- **Intervention evidence:** the Phase 0 J-coordinate positive control passed. No valid Phase 3 causal intervention exists.
- **Observational evidence:** no v2 natural-collision result exists.
- **Statistical evidence:** the readout and positive-control CIs use 10,000 prompt-clustered bootstrap resamples. No downstream significance test was run.
- **Practical magnitude:** pass@10 and intended-answer log-odds are reported above. E_R, E_J, eta, rollout accuracy, and fidelity are unavailable.

## Interpretation boundary

H1, H2, and H3 remain unresolved. The result is not a negative finding about J-space
closure; it is a failure of the preregistered sparse reconstruction/restoration method to
produce acceptable checkpoint states at any candidate layer. The strongest permitted
classification remains D under all preregistered nearby thresholds used for the formal
gate, because there are no eligible layers and no formal downstream trials.

## Reproducibility

The v1 records and `reports/PHASE0_VALIDATION.md` remain unchanged. The v2 freeze,
fresh records, calibration attempts (including invalid rows), failure manifests, processed
summaries, and figures are committed. Figures 1, 2, 12, and 14 visualize measured data;
the remaining required figures are explicitly machine-sourced gated-status panels, not
quantitative results.

### Recorded downstream commands

- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/validate_lens_v2.py --config configs/phase0_v2_confirmatory.yaml`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/calibrate_layers.py --config configs/phase0_v2_confirmatory.yaml`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/closure.py --config configs/pilot_v2.yaml --limit 1`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/natural_collisions.py --config configs/confirm_v2.yaml --limit 1`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/memory_order.py --config configs/confirm_v2.yaml --limit 1 --epochs 1 --budget 1000000`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/distill_controller.py --config configs/confirm_v2.yaml --epochs 1`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/dictionary_sensitivity.py --config configs/confirm_v2.yaml`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/token_time_closure.py --config configs/confirm_v2.yaml --limit 1`
- `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/modularity.py --config configs/confirm_v2.yaml`

## Exploratory protocol v3 update

The v1/v2 records, thresholds, reports, and 0/1400 calibration result remain
byte-identical under the committed SHA-256 regression guard.

- Geometry status: **COMPLETED**
- Pareto status: **COMPLETED**
- V3 clamp calibration: **COMPLETED**
- Behavioral protocols authorized: **none**
- Strongest warranted classification after v3: **D**

Dense state-definition feasibility warning: the median local rank at 1e-4 is 2557/2560 and the median tangent-null dimension is 2. The normalized dense profile is therefore operationally near-injective under the frozen rule. This is not compact H1 evidence and triggers low-dimensional search.

Failed v3 runs are evidence about execution only and are not interpreted as
model behavior:

- `clamp-v3-calibration-20260830T022757Z-3b3fa742-s20260828`: RuntimeError: clamp calibration shards use different configs
- `geometry-v3-20260828T165452Z-1bf9a00a-s20260828`: OSError: We couldn't connect to 'https://hf-mirror.com' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
- `geometry-v3-20260828T165606Z-1bf9a00a-s20260828`: RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'
- `geometry-v3-20260829T135126Z-1bf9a00a-s20260828-pareto-preflight-001`: KeyboardInterrupt: run cancelled
- `geometry-v3-20260829T143434Z-3b3fa742-s20260828-pareto-shard-000`: KeyboardInterrupt: run cancelled
- `geometry-v3-20260829T143434Z-efb45693-s20260828-pareto-shard-001`: KeyboardInterrupt: parent and shard were explicitly cancelled for a measured performance-path correction before any Pareto part was written
- `geometry-v3-20260829T152136Z-3b3fa742-s20260828-pareto-shard-000`: KeyboardInterrupt: run cancelled
- `geometry-v3-20260829T152136Z-efb45693-s20260828-pareto-shard-001`: KeyboardInterrupt: run cancelled
- `geometry-v3-20260829T153732Z-3b3fa742-s20260828-pareto-shard-000`: KeyboardInterrupt: run cancelled
- `geometry-v3-20260829T153732Z-efb45693-s20260828-pareto-shard-001`: KeyboardInterrupt: run cancelled
- `geometry-v3-20260829T155649Z-3b3fa742-s20260828-pareto-shard-000`: KeyboardInterrupt: run cancelled
- `geometry-v3-20260829T155649Z-efb45693-s20260828-pareto-shard-001`: KeyboardInterrupt: run cancelled
- `geometry-v3-20260829T160152Z-3b3fa742-s20260828-pareto-shard-000`: KeyboardInterrupt: run cancelled
- `geometry-v3-20260829T160152Z-efb45693-s20260828-pareto-shard-001`: KeyboardInterrupt: run cancelled
- `geometry-v3-20260829T163042Z-3b3fa742-s20260828-pareto-shard-000`: KeyboardInterrupt: run cancelled
- `geometry-v3-20260829T163042Z-efb45693-s20260828-pareto-shard-001`: KeyboardInterrupt: run cancelled

No H1-Dense, H1-Sparse, H2, or H3 claim is permitted unless a frozen operational
state passes calibration and the paired behavioral, mediation, rollout, and
causal-fidelity gates. Small-perturbation records below 0.20 cannot support those
claims.
