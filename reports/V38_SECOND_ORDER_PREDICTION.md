# V38 — Second Order Prediction

Predictions were sealed before R111.

|model|role|add err|add cos|2nd err|2nd cos|improve|family passes|
|---|---|---|---|---|---|---|---|
|Q|development|0.263|0.971|0.161|0.988|0.330|{'additive': 1, 'higher_order': 0, 'pairwise': 3}|
|Q|validation|0.253|0.972|0.184|0.983|0.297|{'additive': 1, 'higher_order': 0, 'pairwise': 2}|
|F|development|0.656|0.864|0.602|0.833|0.059|{'additive': 0, 'higher_order': 5, 'pairwise': 0}|
|F|validation|0.531|0.885|0.496|0.881|0.055|{'additive': 0, 'higher_order': 5, 'pairwise': 0}|

Q second-order errors are low, but the ≥30% improvement and ≥4-family pairwise gate is not met (family counts 3/5 development, 2/5 validation). F remains well above 0.30 relative error. Thus no model satisfies the formal pairwise class in both roles.
