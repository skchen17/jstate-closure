# Residual Localization Audit v10

<!-- V10-AMENDMENT:START -->
## Corrected estimand and answers

Residual-estimand amendment freeze: `f5b2e0654f389740058dadb9bd17951d8c623fe7f9352b401f381e29cd2d9cf7`. The pre-fix derived result remains archived; no v1-v9 result was changed.

1. **No.** The v9 localization and corrected joint comparator were not mathematically identical: v9 appended non-residualized block coordinates that duplicated compact information.
2. v10 residualizes each architecture block out-of-fold on train, uses train-only held-out residualization, freezes the compact base prediction, and predicts only the OOF target residual. This prevents coefficient refitting/regularization changes from being counted as information.
3. At 512D, the exact same-599D reference gain is `-0.000691` with 95% CI `[-0.001450, 0.000060]`; its maximum discrepancy from the unified comparator is `0.0`. Rich block-space gains are recurrent `0.1600`, conv `0.1605`, KV `0.1309`, and joint `0.1673`.
4. Recurrent and conv are individually strongest and nearly tied; KV is smaller but non-zero. The strongly negative additive interaction is redundancy/non-additivity, not a unique negative information channel.
5. A joint residual near zero requires the same information space and estimand. It is reproduced by `combined_reference`; it does **not** force the richer 3x599 block-specific space to zero. The non-zero architecture-joint result shows that the 599D combined ceiling is lossy with respect to block-resolved state.
<!-- V10-AMENDMENT:END -->


Base freeze: `cf1f153f10f2aa9d522beb44ee6c0b7cb2df2d0b2353db72bcfa5d18eb0d3dc5`; candidate freeze: `3e2368a3b21738f7c00e8ca56290c03d228e978634e31491087bf7507da8a235`.

The v9 architecture-localization comparator was not mathematically identical to the corrected joint comparator. It appended raw block scores after compact coordinates and therefore retained compact information inside each block. v10 uses five-fold out-of-fold residualization on training rows and train-only residualization on held-out rows.

`combined_reference` is the corrected residual inside the same 599D combined ceiling. `architecture_joint` uses the richer 3x599 block-specific score space, so it may remain non-zero even when the combined-reference residual is near zero.

|   dimension | source             |   conditional_gain |       lower |        upper |
|------------:|:-------------------|-------------------:|------------:|-------------:|
|         384 | combined_reference |       -0.000460474 | -0.00197054 |  0.000998679 |
|         384 | recurrent          |        0.158192    |  0.139338   |  0.178292    |
|         384 | conv               |        0.158827    |  0.140449   |  0.178456    |
|         384 | kv                 |        0.131562    |  0.11589    |  0.148377    |
|         384 | architecture_joint |        0.16504     |  0.147038   |  0.184277    |
|         384 | interaction        |       -0.283542    | -0.319394   | -0.249983    |
|         512 | combined_reference |       -0.000690694 | -0.00145034 |  5.96548e-05 |
|         512 | recurrent          |        0.160046    |  0.140816   |  0.1803      |
|         512 | conv               |        0.160529    |  0.141831   |  0.180287    |
|         512 | kv                 |        0.130874    |  0.11538    |  0.147434    |
|         512 | architecture_joint |        0.16727     |  0.149086   |  0.186447    |
|         512 | interaction        |       -0.284179    | -0.319684   | -0.250628    |
