# V20 empirical operator dimension

The oracle coordinate is centered SVD of **positive train-action response fingerprints only**, fitted on training operator states. Held-out action responses never enter the coordinate. The frozen sweep tests k=2,4,8,16,32,64,128, subject to empirical rank, with five continuous-action decoders.

| k | train explained fraction | train fingerprint rel L2 | validation train-action rel L2 |
|---|---|---|---|
| 2 | 0.4867 | 0.5201 | 0.5290 |
| 4 | 0.6457 | 0.4321 | 0.4447 |
| 8 | 0.7829 | 0.3383 | 0.3547 |
| 16 | 0.8728 | 0.2589 | 0.2765 |
| 32 | 0.9288 | 0.1936 | 0.2168 |
| 64 | 0.9647 | 0.1364 | 0.1685 |
| 128 | 0.9860 | 0.0857 | 0.1266 |

Empirical train fingerprint rank is 600. This rank curve describes the 12-action inference fingerprint, not global dimension of P or of all possible actions. A low reconstruction error here alone is insufficient for cross-action compactness.

| model | k | seen rel L2 | unseen direction rel L2 | unseen sign rel L2 | all preliminary gates |
|---|---|---|---|---|---|
| response_svd | 2 | 0.8741 | 0.9358 | 1.0028 | False |
| reduced_rank_regression | 2 | 0.9054 | 0.9485 | 1.0170 | False |
| tucker_factorization | 2 | 0.9074 | 0.9446 | 1.0173 | False |
| bilinear_latent_operator | 2 | 0.8741 | 0.9358 | 1.0028 | False |
| small_nonlinear_latent_operator | 2 | 0.4741 | 1.7701 | 1.2508 | False |
| response_svd | 4 | 0.8584 | 0.9310 | 1.0047 | False |
| reduced_rank_regression | 4 | 0.8809 | 0.9315 | 1.0122 | False |
| tucker_factorization | 4 | 0.8828 | 0.9308 | 1.0148 | False |
| bilinear_latent_operator | 4 | 0.8584 | 0.9310 | 1.0047 | False |
| small_nonlinear_latent_operator | 4 | 0.3706 | 1.5973 | 1.2491 | False |
| response_svd | 8 | 0.8451 | 0.9305 | 1.0056 | False |
| reduced_rank_regression | 8 | 0.8571 | 0.9314 | 1.0094 | False |
| tucker_factorization | 8 | 0.8586 | 0.9302 | 1.0092 | False |
| bilinear_latent_operator | 8 | 0.8451 | 0.9305 | 1.0056 | False |
| small_nonlinear_latent_operator | 8 | 0.3175 | 1.4824 | 1.2032 | False |
| response_svd | 16 | 0.8351 | 0.9295 | 1.0060 | False |
| reduced_rank_regression | 16 | 0.8406 | 0.9306 | 1.0073 | False |
| tucker_factorization | 16 | 0.8412 | 0.9303 | 1.0071 | False |
| bilinear_latent_operator | 16 | 0.8351 | 0.9295 | 1.0060 | False |
| small_nonlinear_latent_operator | 16 | 0.2721 | 1.5075 | 1.1981 | False |
| response_svd | 32 | 0.8294 | 0.9285 | 1.0056 | False |
| reduced_rank_regression | 32 | 0.8310 | 0.9290 | 1.0060 | False |
| tucker_factorization | 32 | 0.8315 | 0.9290 | 1.0057 | False |
| bilinear_latent_operator | 32 | 0.8294 | 0.9285 | 1.0056 | False |
| small_nonlinear_latent_operator | 32 | 0.2356 | 1.5354 | 1.1751 | False |
| response_svd | 64 | 0.8259 | 0.9277 | 1.0057 | False |
| reduced_rank_regression | 64 | 0.8263 | 0.9277 | 1.0057 | False |
| tucker_factorization | 64 | 0.8264 | 0.9277 | 1.0057 | False |
| bilinear_latent_operator | 64 | 0.8259 | 0.9277 | 1.0057 | False |
| small_nonlinear_latent_operator | 64 | 0.2259 | 1.4317 | 1.1751 | False |
| response_svd | 128 | 0.8236 | 0.9273 | 1.0056 | False |
| reduced_rank_regression | 128 | 0.8237 | 0.9273 | 1.0057 | False |
| tucker_factorization | 128 | 0.8242 | 0.9274 | 1.0057 | False |
| bilinear_latent_operator | 128 | 0.8236 | 0.9273 | 1.0056 | False |
| small_nonlinear_latent_operator | 128 | 0.2337 | 1.3318 | 1.1587 | False |

Frozen gates: held-out J and full-stack median cosine ≥ 0.9, relative L2 ≤ 0.3, norm ratio in [0.8,1.2], each-family L2 ≤ 0.35, plus sign/scale/composition diagnostics. `k_operator_min` = **not identified**.

## Interpretation note

The reported numerical rank 600 uses a very small nonzero-singular-value tolerance and is **not** an intrinsic dimension estimate. The energy/reconstruction curve and cross-action held-out errors are the relevant empirical diagnostics. In particular, the nonlinear k=128 model attains seen-direction/new-state relative L2 **0.2337** while its unseen-direction L2 is **1.3318**: this is an action-specific fit, not a cross-action operator state. The poor generalization of this frozen action descriptor/model family cannot prove that no other compact operator parameterization exists.
