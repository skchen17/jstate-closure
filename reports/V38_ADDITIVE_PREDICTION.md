# V38 — Additive Prediction

Prospective additive prediction of R111 from unopened singles:

|model|role|add err|add cos|2nd err|2nd cos|improve|family passes|
|---|---|---|---|---|---|---|---|
|Q|development|0.263|0.971|0.161|0.988|0.330|{'additive': 1, 'higher_order': 0, 'pairwise': 3}|
|Q|validation|0.253|0.972|0.184|0.983|0.297|{'additive': 1, 'higher_order': 0, 'pairwise': 2}|
|F|development|0.656|0.864|0.602|0.833|0.059|{'additive': 0, 'higher_order': 5, 'pairwise': 0}|
|F|validation|0.531|0.885|0.496|0.881|0.055|{'additive': 0, 'higher_order': 5, 'pairwise': 0}|

Q has low median additive error but material second-order improvement, so it does not pass the registered additive class. F additive error/cosine fail directly. Bootstrap CIs are stored per model/role in `trajectory_analysis_*_v38.json`.
