# V18 history versus current snapshot

Train-only PCA compresses 2- and 4-position pre-action J trajectories to the same 128-dimensional budget as current-state comparators. No prior intervention action exists at this control point.

| h | history positions | history over current J | raw over current J | history after raw |
|---|---|---|---|---|
| 1 | 2 | -0.0182 | -0.0014 | -0.0137 |
| 1 | 4 | -0.0100 | -0.0014 | -0.0207 |
| 2 | 2 | -0.0303 | -0.0055 | -0.0362 |
| 2 | 4 | -0.0263 | -0.0055 | -0.0058 |
| 4 | 2 | -0.0165 | -0.0188 | -0.0072 |
| 4 | 4 | -0.0093 | -0.0188 | -0.0207 |
| 8 | 2 | -0.0182 | 0.0061 | -0.0360 |
| 8 | 4 | -0.0202 | 0.0061 | -0.0116 |

Formal last-4-history incremental gain over the current J+raw snapshot:

| h | stack gain | state-bootstrap CI95 | all-family nonnegative |
|---|---|---|---|
| 1 | -0.0207 | [-0.04323030644198335, 0.004123860856117993] | False |
| 2 | -0.0058 | [-0.03440744436088552, 0.0166718965384896] | False |
| 4 | -0.0207 | [-0.05316714140647185, 0.009965607071659444] | False |
| 8 | -0.0116 | [-0.03316916810774019, 0.010598914145873625] | False |

Material history gate: **False**. Full family-wise gains are preserved in `results/v18/processed/v18_adjudication.json`.
A history advantage would indicate limits of this current-snapshot representation or model class, not uniquely missing physical memory fields.
