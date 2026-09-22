# Layerwise State Commit — V28

Exact per-layer REC/Conv old-state restoration and cumulative prefix/suffix restoration were run on 5 states per role. All `240` interventions passed bitwise writeback. Largest single-layer median h1 Q was `33.070516` development and `33.995578` validation; full cumulative medians were `39.667229` and `37.772030`.

The scan is descriptive, not a pre-validated earliest causal layer or exclusive route; strong nonlinear effects permit multiple layers to have large marginals.
