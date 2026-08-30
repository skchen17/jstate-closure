# Token-time closure report

## Status

GATED / NOT EXECUTED. Fresh official-compatible pass@10 was 0.775862 for multihop (58 items) and 0.716797 for order of operations (256 items). The item-clustered MRR advantage was 0.135202, 95% CI [0.109490, 0.161532]. At frozen layer 24, the intended-answer log-odds effect was 3.064270, 95% CI [1.978771, 4.221590], above the 0.0001 null envelope.

Independent closure-layer calibration then tested layers 23, 24, 25, 26, 27, 28, 29. It obtained 0/1400 strictly valid clamp trials; the best layer-level valid rate was 0.000, below the frozen 0.80 requirement. All candidate layers therefore failed only the clamp-valid-rate criterion, and the eligible set is empty.

T1, T2, and T3 macrostate construction and autonomous feedback code are implemented
and unit-tested, but no teacher traces, token-time predictors, recurrent memory models,
or autonomous rollouts were trained. No future teacher token is consumed by the rollout
interface. There is therefore no result about token-time closure, short-memory sufficiency,
controller size, procedural generalization, or intervention fidelity.

## Exploratory protocol v3 low-dimensional authorization

The screen used 1536 fit and 1536 audit transitions. Persistence cosine was 0.965006; the remainder oracle reached 0.990099. Candidates closing at least 80% of that gap: 0. Compact state authorized: False.

The frozen v3 clamp calibration authorized no behavioral protocol, and the
low-dimensional screen authorized no compact operational state. T1/T2/T3 trace
training, autonomous rollout, intervention fidelity, and the controller
parameter sweep therefore remain gated and were not executed.

| Candidate | Dimension | next-state cosine | oracle gap closed | reconstruction cosine |
|:---|---:|---:|---:|---:|
| constrained_learned_encoder | 128 | 0.906758 | -2.321267 | 0.260293 |
| constrained_learned_encoder | 256 | 0.963242 | -0.070291 | 0.552953 |
| constrained_learned_encoder | 512 | 0.977593 | 0.501637 | 0.890273 |
| dense_profile_pca | 32 | 0.958106 | -0.274976 | 0.965793 |
| dense_profile_pca | 64 | 0.969552 | 0.181185 | 0.978516 |
| dense_profile_pca | 128 | 0.976989 | 0.477565 | 0.988009 |
| dense_profile_pca | 256 | 0.979518 | 0.578342 | 0.993913 |
| dense_profile_pca | 512 | 0.979902 | 0.593644 | 0.997121 |
| deterministic_concept_clusters | 32 | 0.646327 | -12.699936 | n/a |
| deterministic_concept_clusters | 64 | 0.662438 | -12.057873 | n/a |
| deterministic_concept_clusters | 128 | 0.707864 | -10.247574 | n/a |
| deterministic_concept_clusters | 256 | 0.790692 | -6.946730 | n/a |
| deterministic_concept_clusters | 512 | 0.879980 | -3.388437 | n/a |
| predictive_linear_bottleneck | 32 | 0.961607 | -0.135464 | 0.801617 |
| predictive_linear_bottleneck | 64 | 0.972214 | 0.287268 | 0.840998 |
| predictive_linear_bottleneck | 128 | 0.978081 | 0.521048 | 0.901812 |
| predictive_linear_bottleneck | 256 | 0.979743 | 0.587316 | 0.962627 |
| predictive_linear_bottleneck | 512 | 0.979901 | 0.593617 | 0.982223 |
| sparse_active_atoms | 50 | 0.974742 | 0.388009 | n/a |
