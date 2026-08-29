# Experiment specification

## Hypotheses

- **H1 — controller:** after matching current J-state, non-J changes have
  little effect on future measured-J states or decisions, and a compact autonomous
  controller reproduces the trajectory.
- **H2 — bus:** measured-J-matched remainder changes materially alter later
  measured-J content or decisions; outside computation determines later writes.
- **H3 — augmented state:** instantaneous J-state fails, but a short J-history
  or low-dimensional recurrent memory closes most of the oracle gap.

## Primary estimand

At the final pre-answer token, perturb the residual at layer L0. At a later
validated workspace layer L1, construct

`h_exp = measured_J_k(h_clean) + measured_R_k(h_perturbed)`.

Only trials passing the preregistered J-match, RMS, and remainder-distance
checks enter confirmatory estimates. The primary output effect is full-vocabulary
Jensen-Shannon divergence from the clean run. The normalized effect
`eta = E_R / (E_J + epsilon)` is reported only when the matched J positive
control clears the numerical-null gate.

## Mediation interpretation

- Little single-clamp effect: evidence for approximate instantaneous closure.
- Single-clamp effect removed by persistent clamps: non-J computation changes
  behavior by writing later information into measured J-space.
- Effect surviving persistent clamps: output-relevant pathways bypass measured
  J-space, which is stronger evidence against J-state sufficiency.

Final-token perturbations cannot change earlier positions under causal masking.
They therefore include an all-position equivalence sanity check. A separate
sequence-state arm perturbs every non-padding reasoning position, restores
measured J at all such positions at L1, then compares persistent final-only and
all-position clamps. Only that arm can identify mediation through other token
positions.

## Phase 0 protocol versions

- v1 is immutable and retains its original failed strict-all-layers gate.
- v2 uses equal item weighting, best synonym and best layer pass@k, all valid
  positions as the main analysis, and copy/position sensitivities.
- The official lens-quality band is selected on old calibration data and frozen
  before fresh confirmation. Individual closure layers then face separate
  readout, positive-control, clamp-validity, and hook-sanity calibration.
- Finite 4,096/8,192/16,384 direction decompositions are called measured-J and
  measured-J remainder. `E_R(M)` is evaluated on paired trials.

## Exploratory protocol v3 geometry contract

Protocol v3 leaves every v1/v2 artifact byte-identical and verifies a committed
SHA-256 manifest before each run. For dictionary sizes 4,096, 8,192, and 16,384,
it distinguishes the unnormalized dense map from the row-normalized sparse
pursuit dictionary. It records direct singular spectra for `A` and centered
`CA`, and analyzes

`J_s(h) = (I - s s^T) C A / ||C A h||`.

Analytic JVP/VJP checks, radial null, numerical/tangent null dimensions,
naturality, and the displacement/equality/RMS Pareto frontier are measurement
preconditions, not behavioral findings. A dense profile that is locally
near-injective is classified as too information-rich for a compact-state claim;
it is never counted as H1 support.

The geometry bank is exactly balanced at 64 unique prompts per family and is
hash-split 32/32 into fit/audit partitions. Direct map spectra cover all seven
layers and all three dictionaries. Sixteen fixed audit states per combination
receive a full local spectrum; the remaining states receive the algebraic rank
implied by the stored centered map and rank-one projection, finite-tolerance interlacing bounds, a
power-iteration top singular estimate, radial residual, and analytic/autograd
JVP/VJP checks. Bounds are labeled `NUMERICALLY_BOUNDED` and are not presented
as full spectra.

The formal Pareto diagnostic uses 16 deterministic, task-balanced audit anchors
(two per family), all nine registered strengths, and all registered null
tolerances. This is an exploratory geometry sample, not the 200-trial clamp
calibration. Runs are layer-sharded across both GPUs and checkpoint each
dictionary/layer part. Limited preflights are separately named and excluded by
the report loader. Local singular vectors are computed once per anchor and
reused across tolerances; normalized sparse dictionaries use an explicitly
tested fast path rather than being normalized again for every pursuit.
Hard-constrained candidates evaluate the train-fit naturality envelope during
each backtracking step, not merely as a post-hoc filter; interrupted runs from
the pre-fix implementation remain preserved as failed manifests.
Sphere-tangent construction treats configured strength as the post-retraction
chord length. It applies a fixed eight-FP32-ulp construction margin so a nominal
0.20 target cannot become 0.1999999 through retraction roundoff; raw achieved
displacement remains stored and the scientific threshold itself is unchanged.

`V3-Dense` requires dense cosine at least 0.995 and top-10 overlap at least 0.8.
`V3-Sparse` independently requires support F1 at least 0.8, weighted Jaccard at
least 0.95, union-aligned coefficient cosine at least 0.995, and reconstruction
cosine at least 0.995. Both require activation RMS drift at most 0.02,
natural-distribution membership, and displacement at least 0.20. Displacements
from 0.05 up to but excluding 0.20 are sensitivity analyses only.

Clamp schedules are serialized before execution. `single` modifies only L1;
`persistent_final` uses the arm's configured L1 positions then only the final
position at later layers; `persistent_all` repeats the L1 scope at every later
selected layer. Records contain the resolved attention-mask-aware positions and
the actual `(layer, position, operation)` edits.

V3 calibration uses one deterministic batch of 200 anchors, balanced as 25 per
task family. A base-trial ID is shared across dictionaries and methods, while a
method-paired ID is shared across dictionaries, so attrition and dictionary-size
effects can be evaluated on exactly matched constructions.
Calibration is layer-hash-sharded across both GPUs, checkpoints every
dictionary/layer cell, and merges only shards with an identical launch group,
commit, config digest, and activation-bank manifest. The 95% naturality gate is
computed over candidates passing state equality/RMS/displacement before the
naturality criterion is applied.
A protocol needs at least 160/200 strict-valid trials per layer, 95% naturality
among valid trials, fewer than 5% numerical optimizer failures, passing hook
controls, and at least four ordered eligible layers. The behavioral runner
refuses to run unless all source/config/data hashes match the committed Phase 3
freeze manifest.

At behavioral time, the initial L1 clamp must pass formal displacement. Later
persistent clamps require measured-state equality, RMS, and naturality but do
not manufacture a new 0.20 displacement when the current state is already near
clean. Dense clamps project the observed current-minus-clean displacement into
the local tangent-null subspace; sparse clamps use the independent sparse
definition. Perturbation stripping and positive controls use the same selected
state definition.

If dense geometry is near-injective or fails the formal displacement gate, the
low-dimensional screen compares sparse active atoms, dense-profile PCA,
deterministic concept aggregation, predictive linear bottlenecks, and a learned
linear encoder constrained to contain the frozen Phase 0/positive-control
concept axes exactly. It selects only the latest completed formal spectrum and
Pareto shards. Prediction screening alone cannot authorize a compact state;
Phase 0 retention and intervention-retention gates remain mandatory.

## Statistical contract

Pilot cells target at least 100 valid interventions; confirmatory task families
target at least 500. Confidence intervals use 10,000 prompt-clustered bootstrap
resamples. Raw effects, valid/attempted counts, attrition, and threshold sweeps
are always reported. Natural collisions remain observational.

## Controller contract

Layer-depth histories and token-time histories are separately labeled. T1 uses
the final eligible layer, T2 pools normalized dense scores across the eligible
band, and T3 adds a 128-dimensional recurrent state. Models receive the
appropriate clock and either true first measured-J state or a separately counted
standalone initializer. Primary evaluation is autonomous rollout with predicted
state fed back. Learned embeddings, initializers, fact encoders, transition
bodies, and heads all count toward the parameter budget. Counterfactual
coordinate swaps are required for causal fidelity.

## Executed protocol-v2 outcome

The frozen fresh Phase 0 v2 gate passed. Independent layer calibration then
tested 200 balanced clamp constructions at each candidate layer 23–29 and found
zero strictly valid clamps at every layer. No layer met the required 80% valid
rate; therefore closure, mediation, dictionary sensitivity, collisions,
layer-depth/token-time prediction, controllers, modularity, and the optional 27B
confirmation are formally gated. The registered thresholds were not changed
after observing this failure. See the machine summaries and generated reports
for exact values and hashes.
