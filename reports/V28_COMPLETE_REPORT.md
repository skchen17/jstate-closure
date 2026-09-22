# V28 Complete Report

## Identity and frozen starting point

**Token-Level Persistent State Read/Write Transaction**. Parent `83235995faa5aefb0f9fd9dab6f865a3506df2e1`; protocol `b4a0b8cc34e3caf98bacb885157c24f23db88cf54c497c8cc5f0da1629d4363f`; boundary `d38905cfdd0e9480d59751a084f578f59f1dc546823c42adf91c32a4b6298d87`; adjudication `583e9895d58f65ee4eb62f93883826260ad407ae72f97ced723dc3e152a13f2d`. V27-F remains the starting point: V26's current silence was post-readout, whereas tested pre-readout perturbations changed current output. V28 does not revive pre-existing current-silent memory, compact state, or H3.

## Architecture and prospective design

In Qwen3.5, six layers have native REC/Conv state and two have attention KV. After the complete natural token-t forward, layer-30 current readout and outgoing cache are both available. Exact incoming REC/Conv fields can replace the natural outgoing fields without touching current activations; KV cannot be shortened to the old sequence length without changing positional semantics, so its new-slot copy is a separately labeled diagnostic. All future branches use identical next tokens.

Disjoint balanced panels were frozen at 25 calibration, 75 development, 50 validation, and 50 independent-final states. One development pilot state was explicitly removed from formal analysis after the layer-23 versus layer-30 J naming ambiguity was discovered; primary results use 74 development states and layer-30 readout-J at both t and t+1. The pilot future observation and unchanged thresholds are recorded in an append-only endpoint amendment.

## Formal outcomes

`V28-A=TRUE`, `V28-B=TRUE`, `V28-C=TRUE`, `V28-D=FALSE`, `V28-E=FALSE`, `V28-F=FALSE`, `V28-G=FALSE`, `V28-H=FALSE`.

The predeclared **exact REC+Conv natural-write block** left the already-computed current readout unchanged (Q=0) while altering next-token readout by median Q `34.551789` development, `36.818478` validation, and `35.872796` independent final. All five families replicated, all formal writebacks were exact, and the potent fraction was 1.0 in each panel. Incoming-state read interventions, by contrast, changed current output. Natural-write dose effects declined monotonically as the retained REC/Conv update approached 100%.

Natural same-prompt donor branches showed exact **full outgoing-cache** transfer under both reciprocal directions, with cosine ≈1 and magnitude ratio 1. This is a complete-cache identity control: making the future cache equal to a donor and replaying the same token should reproduce that donor. It establishes full-state transplant fidelity/transferability but does **not** identify a selective single-channel route. Single REC, Conv, KV, and REC+Conv partial transplants did not pass the reciprocal donor-direction gate. Individual write blocks nevertheless had large marginal effects, so neither a purely distributed-only necessity claim nor a global channel ranking is justified.

## Workspace and horizons

Next-token layer-30 J changes under write block by median Q `89.562832` development and `91.570364` validation despite fixed current J. This supports that next workspace depends causally on the prior outgoing REC/Conv commit, not that it is fully constructed by or identical to a compact state. h1/h2/h4 median Q is `32.265/7.898/3.448` development and `36.671/8.854/3.671` validation.

## Limits and interpretation

The zero current write effect is guaranteed by the after-readout interception boundary; the nontrivial causal result is that blocking **the natural outgoing update**, rather than adding an arbitrary future perturbation, changes subsequent computation and survives dose/replication/final tests. Old-state restoration can be off-manifold; same-family shuffled full cache also causes large future changes. Exact full-cache transfer is expected, and partial transfer failure prevents a localized-carrier claim. Predictive write features, norm-matched random write, and a matched small-write causal control were not established and are excluded from formal gates. No compact or complete-state inference is made.

`H2_REMAINS=TRUE`; `H3_AUTHORIZED=FALSE`; `DYNAMIC_STATE_SEARCH_AUTHORIZED=FALSE`; `AUTONOMOUS_CONTROLLER_AUTHORIZED=FALSE`.

V28 tests: `6 passed in 1.73s`. Full suite: `274 passed, 3 failed, 3 warnings in 51.90s`; three inherited cumulative-report hash checks fail.
