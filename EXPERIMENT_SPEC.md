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
