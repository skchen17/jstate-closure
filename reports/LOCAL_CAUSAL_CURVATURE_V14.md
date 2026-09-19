# Local causal curvature — V14

The top `9` local causal directions were taken from frozen V13 rank-512 JVP matrices, one train anchor per family. Central second differences at radius `1.0`, two mixed-direction pairs, and held-out radii `0.5` and `2.0` were measured. Since the infinitesimal JVP gate failed, these are **finite-scale curvature diagnostics, not validated differential Hessian estimates**.

|   radius | target              | surrogate   |   direction |   relative_l2 |   magnitude |
|---------:|:--------------------|:------------|------------:|--------------:|------------:|
|    0.500 | j                   | linear      |       0.921 |         0.541 |       1.253 |
|    0.500 | j                   | quadratic   |       0.949 |         0.422 |       1.145 |
|    0.500 | logits              | linear      |       0.867 |         0.602 |       1.096 |
|    0.500 | logits              | quadratic   |       0.907 |         0.529 |       1.166 |
|    0.500 | semantic_continuous | linear      |       0.854 |         0.681 |       1.040 |
|    0.500 | semantic_continuous | quadratic   |       0.926 |         0.463 |       1.063 |
|    0.500 | workspace           | linear      |       0.755 |         0.701 |       0.931 |
|    0.500 | workspace           | quadratic   |       0.924 |         0.408 |       1.028 |
|    2.000 | j                   | linear      |       0.643 |         2.289 |       2.595 |
|    2.000 | j                   | quadratic   |       0.838 |         2.253 |       2.878 |
|    2.000 | logits              | linear      |       0.556 |         1.257 |       1.547 |
|    2.000 | logits              | quadratic   |       0.799 |         1.453 |       2.232 |
|    2.000 | semantic_continuous | linear      |       0.535 |         1.389 |       1.561 |
|    2.000 | semantic_continuous | quadratic   |       0.804 |         1.724 |       2.315 |
|    2.000 | workspace           | linear      |       0.172 |         1.118 |       0.640 |
|    2.000 | workspace           | quadratic   |       0.933 |         0.902 |       1.587 |

- First-order valid radius under the all-target direction/magnitude/relative-error rule at tested held-out radii `0.5/2.0`: `None`.
- Second-order valid radius under the same tested radii: `None`.
- The quadratic term improves direction prediction at radius `0.5`, but relative errors remain above `0.20`; it does not validate a finite local surrogate.
- Mixed second-difference terms were retained in `results/v14/processed/local_causal_mixed_curvature_v14.parquet`.
