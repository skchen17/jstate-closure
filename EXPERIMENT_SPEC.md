# Experiment specification

## Hypotheses

- **H1 — controller:** after matching current J-state, non-J changes have
  little effect on future J-states or decisions, and a compact autonomous
  controller reproduces the trajectory.
- **H2 — bus:** J-matched non-J changes materially alter later J-content or
  decisions; non-J computation determines later workspace writes.
- **H3 — augmented state:** instantaneous J-state fails, but a short J-history
  or low-dimensional recurrent memory closes most of the oracle gap.

## Primary estimand

At the final pre-answer token, perturb the residual at layer L0. At a later
validated workspace layer L1, construct

`h_exp = J_k(h_clean) + R_k(h_perturbed)`.

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

## Statistical contract

Pilot cells target at least 100 valid interventions; confirmatory task families
target at least 500. Confidence intervals use 10,000 prompt-clustered bootstrap
resamples. Raw effects, valid/attempted counts, attrition, and threshold sweeps
are always reported. Natural collisions remain observational.

## Controller contract

Layer depth is time. Models receive the layer clock and either true first
J-state or a separately counted standalone initializer. Primary evaluation is
autonomous rollout with predicted state fed back. Learned embeddings, initializers,
fact encoders, transition bodies, and heads all count toward the parameter
budget. Counterfactual J-coordinate swaps are required for causal fidelity.

