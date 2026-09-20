# Finite-response nonlinearity decomposition — V16

All statistics below are medians of measured reliable actions only, with unreliables retained in the bank denominator. `G_even/G_odd` uses the exact ± finite responses; scale terms are descriptive least-squares coefficients over the frozen signed sweep, not proof that dynamics are polynomial.

## Odd/even ratio

| target              |   count |   even_over_odd |
|:--------------------|--------:|----------------:|
| j                   |    4219 |           0.238 |
| logits              |    4219 |           0.303 |
| semantic_continuous |    4219 |           0.353 |
| stacked_normalized  |    4219 |           0.373 |
| workspace           |    4219 |           0.581 |

## Scale linear/quadratic/cubic fit

| target              |   count |   linear_norm |   quadratic_norm |   cubic_norm |   quadratic_over_linear |   scale_fit_relative_l2 |
|:--------------------|--------:|--------------:|-----------------:|-------------:|------------------------:|------------------------:|
| j                   |     120 |         0.074 |            0.016 |        0.009 |                   0.146 |                   0.109 |
| logits              |     120 |         5.666 |            0.953 |        0.577 |                   0.152 |                   0.151 |
| semantic_continuous |     120 |         6.023 |            1.343 |        0.582 |                   0.221 |                   0.154 |
| stacked_normalized  |     120 |         4.390 |            0.972 |        0.443 |                   0.266 |                   0.147 |
| workspace           |     120 |         1.647 |            0.832 |        0.141 |                   0.462 |                   0.121 |

## Pair interactions

`I_ij=G(e_i+e_j)-G(e_i)-G(e_j)`; target-block contribution is reported separately, including J, logits, continuous semantic, workspace and normalized stack.

| target              |   count |   interaction_norm |   interaction_over_sum |
|:--------------------|--------:|-------------------:|-----------------------:|
| j                   |     120 |              0.008 |                  0.109 |
| logits              |     120 |              0.711 |                  0.144 |
| semantic_continuous |     120 |              0.944 |                  0.157 |
| stacked_normalized  |     120 |              0.629 |                  0.141 |
| workspace           |     120 |              0.384 |                  0.148 |

Explicit low-rank REC×Conv, REC×KV and Conv×KV bilinear terms are compared against the ordinary quadratic model on held-out states/pairs. These are channel-energy proxy interactions, not a unique physical decomposition:

|   k | model            |   test_count |   j_relative_l2 |   stacked_normalized_relative_l2 |
|----:|:-----------------|-------------:|----------------:|---------------------------------:|
|   2 | channel_bilinear |           40 |           0.511 |                            0.501 |
|   2 | quadratic        |           40 |           0.511 |                            0.501 |
|   4 | channel_bilinear |          160 |           0.512 |                            0.511 |
|   4 | quadratic        |          160 |           0.512 |                            0.511 |
|   8 | channel_bilinear |          160 |           0.512 |                            0.511 |
|   8 | quadratic        |          160 |           0.512 |                            0.511 |
|  16 | channel_bilinear |          160 |           0.512 |                            0.511 |
|  16 | quadratic        |          160 |           0.512 |                            0.511 |
|  32 | channel_bilinear |          160 |           0.512 |                            0.511 |
|  32 | quadratic        |          160 |           0.512 |                            0.511 |

## Sparse triple Möbius interaction

No estimable records.

Not all pair subsets of the frozen triples were sampled, so unavailable three-way contrasts remain **not estimable**; no zero interaction is imputed. Full per-state values: `/data/CSK/J-space-project/jstate-closure/results/v16/processed/nonlinear_decomposition_v16.parquet`.
