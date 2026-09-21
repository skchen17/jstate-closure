# V21 state × action × decoder factorial

The table keeps S, Z and G comparisons separated. Ridge values were chosen only on held-out development states/actions, then refit on 600 training operator states. G4/G5 architecture (width 64, 25 fixed epochs, seed 2021) was frozen in a separate design stage; the base config's exploratory 128/60 placeholders were not used. Parameters are reported for neural models. Z3 depends on P and therefore does not have an eligible separable G0–G3 kernel under this implementation.

| S | Z | G | seen/new state L2 | unseen direction L2 | unseen sign L2 | unseen direction+sign L2 | parameters |
|---|---|---|---:|---:|---:|---:|---:|
| S1 | Z0 | G0_additive | 0.7826 | 0.9681 | 0.9516 | 1.0562 | kernel/closed-form |
| S1 | Z0 | G1_bilinear | 0.2608 | 2.7229 | 1.5316 | 2.5509 | kernel/closed-form |
| S1 | Z0 | G2_quadratic_action | 0.1342 | 1.2892 | 1.2332 | 1.2338 | kernel/closed-form |
| S1 | Z0 | G3_tensor_polynomial | 0.1388 | 1.3040 | 1.2408 | 1.2325 | kernel/closed-form |
| S2 | Z0 | G0_additive | 0.7827 | 0.9700 | 0.9540 | 1.0610 | kernel/closed-form |
| S2 | Z0 | G1_bilinear | 0.2046 | 3.1384 | 1.8673 | 2.9483 | kernel/closed-form |
| S2 | Z0 | G2_quadratic_action | 0.1360 | 1.3137 | 1.2479 | 1.2392 | kernel/closed-form |
| S2 | Z0 | G3_tensor_polynomial | 0.1276 | 1.3268 | 1.2533 | 1.2519 | kernel/closed-form |
| S2 | Z1 | G0_additive | 0.7733 | 0.8353 | 0.8541 | 0.9524 | kernel/closed-form |
| S2 | Z1 | G1_bilinear | 0.1842 | 0.7248 | 1.0231 | 1.1089 | kernel/closed-form |
| S2 | Z1 | G2_quadratic_action | 0.0615 | 0.6386 | 0.9371 | 0.9282 | kernel/closed-form |
| S2 | Z1 | G3_tensor_polynomial | 0.1053 | 0.6405 | 0.9362 | 0.9282 | kernel/closed-form |
| S2 | Z2 | G0_additive | 0.7183 | 0.9934 | 1.0085 | 1.0574 | kernel/closed-form |
| S2 | Z2 | G1_bilinear | 842432902219.7047 | 645984439655.0625 | 850058333875.1685 | 575421464885.3549 | kernel/closed-form |
| S2 | Z2 | G2_quadratic_action | 3994030481954932969318895942381862912.0000 | 717848777956228734944297369724780544.0000 | 4030183162052610281029100368478339072.0000 | 639435828906734665536668339674284032.0000 | kernel/closed-form |
| S2 | Z2 | G3_tensor_polynomial | 5070498717517716529732284777240723456.0000 | 911128340391791016251472429554073600.0000 | 5116395242068118697553479506761613312.0000 | 811602838187693156001943274276257792.0000 | kernel/closed-form |
| S3 | Z0 | G0_additive | 0.7830 | 0.9699 | 0.9532 | 1.0608 | kernel/closed-form |
| S3 | Z0 | G1_bilinear | 0.2129 | 3.1312 | 1.8561 | 2.9406 | kernel/closed-form |
| S3 | Z0 | G2_quadratic_action | 0.1474 | 1.3132 | 1.2478 | 1.2360 | kernel/closed-form |
| S3 | Z0 | G3_tensor_polynomial | 0.1375 | 1.3256 | 1.2530 | 1.2482 | kernel/closed-form |
| S3 | Z1 | G0_additive | 0.7735 | 0.8355 | 0.8535 | 0.9523 | kernel/closed-form |
| S3 | Z1 | G1_bilinear | 0.1918 | 0.7256 | 1.0181 | 1.1060 | kernel/closed-form |
| S3 | Z1 | G2_quadratic_action | 0.0831 | 0.6393 | 0.9378 | 0.9273 | kernel/closed-form |
| S3 | Z1 | G3_tensor_polynomial | 0.1182 | 0.6411 | 0.9261 | 0.9252 | kernel/closed-form |
| S0 | Z1 | G0_additive | 0.8060 | 0.8556 | 0.8676 | 0.9463 | kernel/closed-form |
| S0 | Z1 | G1_bilinear | 0.6578 | 0.7889 | 0.9523 | 1.0224 | kernel/closed-form |
| S0 | Z1 | G2_quadratic_action | 0.6590 | 0.7565 | 0.9312 | 0.9333 | kernel/closed-form |
| S0 | Z1 | G3_tensor_polynomial | 0.6599 | 0.7584 | 0.9624 | 0.9388 | kernel/closed-form |
| S0 | Z0 | G0_additive | 0.8150 | 0.9879 | 0.9672 | 1.0564 | kernel/closed-form |
| S0 | Z0 | G1_bilinear | 0.6581 | 2.7145 | 1.4851 | 2.5122 | kernel/closed-form |
| S0 | Z0 | G2_quadratic_action | 0.6568 | 1.2292 | 1.1773 | 1.1274 | kernel/closed-form |
| S0 | Z0 | G3_tensor_polynomial | 0.6576 | 1.2389 | 1.1831 | 1.1304 | kernel/closed-form |
| S1 | Z1 | G0_additive | 0.7732 | 0.8328 | 0.8511 | 0.9469 | kernel/closed-form |
| S1 | Z1 | G1_bilinear | 0.2364 | 0.7197 | 0.9759 | 1.0311 | kernel/closed-form |
| S1 | Z1 | G2_quadratic_action | 0.1266 | 0.6409 | 0.9305 | 0.9265 | kernel/closed-form |
| S1 | Z1 | G3_tensor_polynomial | 0.1361 | 0.6433 | 0.9273 | 0.9257 | kernel/closed-form |
| S4 | Z0 | G0_additive | 0.7998 | 0.9743 | 0.9650 | 1.0537 | kernel/closed-form |
| S4 | Z0 | G1_bilinear | 0.4841 | 2.9195 | 1.6581 | 2.7189 | kernel/closed-form |
| S4 | Z0 | G2_quadratic_action | 0.4519 | 1.2815 | 1.2246 | 1.1842 | kernel/closed-form |
| S4 | Z0 | G3_tensor_polynomial | 0.3755 | 1.2911 | 1.2264 | 1.1966 | kernel/closed-form |
| S4 | Z1 | G0_additive | 0.7906 | 0.8397 | 0.8645 | 0.9428 | kernel/closed-form |
| S4 | Z1 | G1_bilinear | 0.4718 | 0.7357 | 0.9922 | 1.0798 | kernel/closed-form |
| S4 | Z1 | G2_quadratic_action | 0.4605 | 0.6847 | 0.9410 | 0.9305 | kernel/closed-form |
| S4 | Z1 | G3_tensor_polynomial | 0.3500 | 0.6636 | 0.9231 | 0.9291 | kernel/closed-form |
| S2 | Z2_train_floor_corrected | G0_additive | 0.7732 | 0.8338 | 0.8542 | 0.9516 | kernel/closed-form |
| S2 | Z2_train_floor_corrected | G1_bilinear | 0.1839 | 0.7247 | 1.0235 | 1.1080 | kernel/closed-form |
| S2 | Z2_train_floor_corrected | G2_quadratic_action | 0.0614 | 0.6329 | 0.9125 | 0.9261 | kernel/closed-form |
| S2 | Z2_train_floor_corrected | G3_tensor_polynomial | 0.1053 | 0.6348 | 0.9119 | 0.9261 | kernel/closed-form |
| S2 | Z3 | G4_joint_MLP | 0.3734 | 0.7725 | 0.7517 | 0.9564 | 246624 |
| S2 | Z3 | G5_rank8_hypernetwork | 0.3431 | 0.6983 | 0.7227 | 0.7850 | 371320 |
| S4 | Z3 | G4_joint_MLP | 0.5235 | 0.7933 | 0.7896 | 0.8729 | 31392 |
| S4 | Z3 | G5_rank8_hypernetwork | 0.5027 | 0.7038 | 0.7851 | 0.7892 | 156088 |
| S1 | Z0 | G4_joint_MLP | 0.4533 | 1.5490 | 1.2362 | 1.0647 | 32160 |
| S1 | Z0 | G5_rank8_hypernetwork | 0.4189 | 3.1450 | 1.1668 | 2.9455 | 158144 |
| S2 | Z0 | G4_joint_MLP | 0.3734 | 1.7455 | 1.2192 | 1.1237 | 245152 |
| S2 | Z0 | G5_rank8_hypernetwork | 0.3848 | 3.5292 | 1.1561 | 3.2775 | 371136 |
| S2 | Z1 | G4_joint_MLP | 0.3736 | 0.7747 | 0.8125 | 0.9557 | 244896 |
| S2 | Z1 | G5_rank8_hypernetwork | 0.3350 | 0.8317 | 1.1604 | 1.3966 | 371104 |
| S2 | Z2 | G4_joint_MLP | 0.3770 | 0.7969 | 0.7979 | 0.9405 | 244896 |
| S2 | Z2 | G5_rank8_hypernetwork | 0.3341 | 0.8254 | 1.1605 | 1.3917 | 371104 |
| S3 | Z1 | G4_joint_MLP | 0.3766 | 0.7682 | 0.7948 | 0.9469 | 466080 |
| S3 | Z1 | G5_rank8_hypernetwork | 0.3347 | 0.8334 | 1.1565 | 1.4000 | 592288 |
| S0 | Z1 | G4_joint_MLP | 0.6749 | 0.8033 | 0.8410 | 0.9612 | 285856 |
| S0 | Z1 | G5_rank8_hypernetwork | 0.6710 | 0.8524 | 1.0497 | 1.2281 | 412064 |

Additional reliability-qualified finite tests for the representative near-best S2×Z1×G2 candidate on 50 P0/Pq validation operator states:

| heldout action type | relative L2 | qualified states |
|---|---:|---:|
| unseen_amplitude_seen_direction | 0.5118 | 50 |
| unseen_amplitude_unseen_direction | 0.6733 | 50 |
| unseen_dense_combination | 0.6239 | 50 |
| unseen_pair | 0.6512 | 50 |

The isolated action comparison S2×Z0×G2 → S2×Z1×G2 improves unseen direction by 0.6751, while S1→S2 at fixed Z1/G2 improves only 0.0023. Stronger G4/G5 models do not pass the strict gate. The original unnormalized Z2 numerical failure remains in `bottleneck_factorial_v21.json`; only `z2_channel_normalization_correction_v21.json` is used for scientific comparison.
