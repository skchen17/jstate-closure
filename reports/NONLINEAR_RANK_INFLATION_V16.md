# Nonlinear finite-rank inflation — V16

V15's restricted actual r95≈14 versus exact-JVP r95≈7 is a **linear response spectrum**, not 14 independent causal degrees of freedom. V16 observed positive-single finite spectra on new validation states are:

|      k |   states |   r90 |   r95 |    r99 |
|-------:|---------:|------:|------:|-------:|
|  2.000 |   50.000 | 1.000 | 1.000 |  1.000 |
|  4.000 |   50.000 | 2.000 | 3.000 |  3.000 |
|  8.000 |   50.000 | 4.000 | 5.000 |  6.000 |
| 16.000 |   50.000 | 5.000 | 6.000 |  9.000 |
| 32.000 |   50.000 | 6.000 | 8.000 | 14.000 |

These spectra center a nested set of up to 32 positive single actions per state; V15's 512-probe signed-central-response spectrum uses a different distribution, so the two r95 values are not a direct replication comparison.

No nonlinear model met all frozen J and stacked-response heldouts. Thus predicted linear-only, quadratic and full-model spectra are **not validated explanations** of observed rank inflation, and the phrase `FINITE_LINEAR_RANK_INFLATION_EXPLAINED_BY_LOW_DIMENSIONAL_NONLINEAR_ACTIONS` is not licensed. Per-state spectra, including first 40 singular values, are in `results/v16/processed/v16_analysis.json`. Variance explained by any unvalidated model would be training/descriptive only.
