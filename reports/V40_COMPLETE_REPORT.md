# V40 — Complete Report

Functional and Temporal Decomposition of Persistent State in Hybrid Language Models

V40 is execution-complete across its sealed Qwen functional, temporal, and trajectory branches and Falcon functional comparison.
V39 and the original V40 calibration are unchanged historical inputs, not formal V40 outcomes.

## Frozen design and inference scope

The state is the independent unit. Task labels were generated before model execution; four state-ID-disjoint pools contain calibration=16, development=64, validation=32, independent_final=32 states.
Qwen is primary and Falcon is descriptive comparison. The fixed readout is after_one_space; uncertainty uses 2000 state-unit bootstrap resamples and 95% percentile intervals.
Formal function, delayed continuation, and REC-depth factorial outcomes are reported separately; controls and invalid/unavailable rows remain visible. The answer-alphabet vector metric is secondary and exploratory, not a substitute for externally labelled recovery.

## Scientific assessment

- Qwen independent-final task-grounded REC+Conv donor-recovery advantage over both matched mismatch/shuffled controls, positive 95% lower bounds: **True**.
- Qwen independent-final task-grounded positive REC three-way depth interaction at both tested delays: **False**.
- Channel-specific empirical delay profiles and any identifiable observed-grid half-crossings are in `V40_TEMPORAL_PERSISTENCE.md`; no universal decay law or semantic storage role is inferred.
- Falcon remains a comparison only where its native paired-answer baseline is interpretable; no shared or model-specific ontology is assumed from an inadequate assay.

The broad V40 hypothesis succeeds only if channel roles, temporal specialization, trajectory causality, and external task grounding are each supported. A partial or negative result is a completed experiment, not proof of the untested claims.
See the five topic reports and `V40_ALL_REPORTS.md` for the full version-local account.

## Answer to the V40 scientific question

**Functional content under this assay.** Qwen completed 32/32 states; 19 were native-pair-qualified and the frozen Qwen gate was True. Qualified donor-answer recovery: REC_only=0/19, Conv_only=3/19, KV_only=15/19, REC_Conv=5/19, REC_Conv_KV=19/19. This is a causal channel profile on generated tasks, not a proof that a tensor stores a semantic variable.

**Task grounding.** REC+Conv versus matched same-prefix mismatch: 0.267 [0.067, 0.468]; versus shuffled natural REC+Conv: 0.263 [0.105, 0.474]. The incremental REC+Conv-versus-Conv recovery contrast is 0.105 [0.000, 0.263]; a CI including zero does not establish a separate REC answer contribution.

**Temporal persistence.** Qwen native-qualified recovery by delay: REC_only 0/19 at delay 0 and 0/14 at delay 32, Conv_only 3/19 at delay 0 and 0/14 at delay 32, KV_only 15/19 at delay 0 and 12/14 at delay 32, REC_Conv 5/19 at delay 0 and 0/14 at delay 32. These per-delay denominators change, so the rates alone are not a same-cohort decay curve. Common native-qualified states across all delays: 12; observed-grid half-crossings: Conv_only=not identifiable, KV_only=not identifiable, REC_Conv=not identifiable, REC_Conv_KV=not identifiable, REC_only=not identifiable. No continuous or universal half-life is inferred.

**Depth trajectory.** Three-way external-answer margin interaction at delay 0: -0.018 [-0.056, 0.021] (n=19); at delay 8: -0.017 [-0.057, 0.020] (n=22). The predeclared gates were False and False. Donor-like hidden geometry without answer recovery is not mediation.

**Cross-model limit.** Falcon completed 32/32 states but only 8 were native-pair-qualified under this readout; its channel counts are descriptive and cannot establish a common architecture-independent role.

**Overall.** V40 execution is complete, but its full four-part scientific success criterion is not established because the task-grounded depth-trajectory gate failed. Any positive functional and empirical temporal profiles remain bounded to these generated tasks, model versions, readout, delays, and tested state pool.
