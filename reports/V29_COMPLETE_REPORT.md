# V29 Complete Report

## Identity and frozen history

**Causal Content of Natural Persistent-State Writes — What Does a Token Commit to the Future?** Parent `8d3954ef05957ab22f27261ea7eaa217497b2b62`; base protocol `86389aa651802da41d09c0217c9f588586eab45a7274aee4d5d1ae42c2114913`; adjudication `f404213f48100effb31d7aa99dfc5141faacb7dfdb3eca83f47c67f0802b858a`. V28's exact REC+Conv write block and full-cache identity control remain historical. V29 does not reinterpret V28 failed partial transfers as successful carriers.

## Prospective same-incoming fork

The frozen 25/75/50/50 balanced panels exclude V28 IDs. At each prompt prefix, top-plausible distinct-surface token A/B were selected from the model distribution **before** their writes or future responses were observed. Both branches start with the identical native cache and complete their current-token computation naturally. The next input token is the same frozen V18 teacher token. Current readout and outgoing state are captured before any transplant. Same-token calibration replay was exact.

An explicit interface amendment preserves a failed preliminary calibration: reusing V28's posterior 6 REC/Conv + 2 KV layers as a purported full V29 cache yielded median relative L2 `0.823`. A transient 53-state development run was interrupted before formal partition output. No threshold, token pair, state role, or finalist changed. Formal V29 covers **all 24 recurrent/conv layers and 8 attention layers**; full cache transplant is bitwise donor-equal and response-exact.

## Causal results

Natural token forks generated distinct writes and shared-next-token future Q median `90.529` development, `92.302` validation, `92.745` independent final. All exact native-field writebacks passed. Full-cache transfer is an identity ceiling, not carrier localization.

| channel | role | A←B cosine / magnitude / rel-L2 | B←A cosine / magnitude / rel-L2 | reciprocal gate |
|---|---|---:|---:|---:|
| REC | development | 0.062 / 0.095 / 0.999 | 0.259 / 0.132 / 0.980 | False |
| REC | validation | -0.002 / 0.095 / 1.003 | 0.358 / 0.137 / 0.962 | False |
| Conv | development | 0.988 / 0.971 / 0.163 | 0.992 / 0.989 / 0.127 | True |
| Conv | validation | 0.987 / 0.962 / 0.175 | 0.992 / 0.993 / 0.125 | True |
| KV | development | 0.082 / 0.067 / 0.997 | 0.108 / 0.067 / 0.995 | False |
| KV | validation | 0.088 / 0.067 / 0.996 | 0.093 / 0.070 / 0.996 | False |
| REC+Conv | development | 0.998 / 0.995 / 0.067 | 0.998 / 0.997 / 0.067 | True |
| REC+Conv | validation | 0.998 / 0.996 / 0.070 | 0.998 / 0.996 / 0.067 | True |

Conv-only and REC+Conv donor writes pass reciprocal development and validation gates; REC-only and KV-only fail. The **predeclared REC+Conv** finalist passed the 50-state independent final (`A←B` median cosine `0.998`, `B←A` `0.997`). KV remained recipient-native in this transfer. Thus the tested natural write commits future-relevant, token-conditioned recurrent/conv content beyond exact KV token-history storage. Conv alone passes, so a strict distributed-only/no-single-channel claim is false. These experiments do not decode a semantic variable inside a channel.

Ten states per role tested four pre-write next-token probes: REC+Conv median donor cosine was `0.988` development and `0.986` validation (A←B), with full cache exact. Native REC+Conv greatly outperformed norm-related random, shuffled and sign-flipped controls. h1/h2/h4 donor-specific effects were measured only after h1 gates. Cross-state factorials found aligned but imperfectly transferable write contrasts; neither global nor state-dependent code is formally settled.

## Workspace, dimension, and limits

Held-out diagnostic R² for token+J future prediction was `0.423`; adding coarse write geometry gave `0.488`. This predictive gain is **not** a J-matched causal proof of workspace incompleteness. Current J is a readout, not a persistent cache field.

Train-only REC+Conv write-effect PCA tested k `[4, 8, 16, 32, 64]` on held-out development and validation; qualified k: `32, 64`. k `[128, 256]` exceeded the train span and were not tested. Any qualifying k would be a write-effect approximation, **not** a complete or dynamical state dimension. No arbitrary perturbation is promoted to a natural write.

Formal outcomes:
- `V29_A_TOKEN_CONDITIONED_STATE_WRITE_CONFIRMED`: **True**
- `V29_B_SAME_BACKGROUND_WRITE_TRANSFER_CONFIRMED`: **True**
- `V29_C_RECURRENT_WRITE_CARRIES_NONTRIVIAL_CONTENT`: **True**
- `V29_D_KV_DOMINATED_TOKEN_CARRYOVER`: **False**
- `V29_E_DISTRIBUTED_WRITE_CONTENT`: **False**
- `V29_F_STATE_DEPENDENT_WRITE_CODE`: **False**
- `V29_G_GLOBAL_WRITE_CODE`: **False**
- `V29_H_CURRENT_WORKSPACE_INCOMPLETE_FOR_WRITE`: **False**
- `V29_I_COMPACT_WRITE_EFFECT_DIMENSION_IDENTIFIED`: **True**
- `V29_J_NO_COMPACT_WRITE_EFFECT_DIMENSION_IDENTIFIED`: **False**
- `V29_K_WRITE_CONTENT_REMAINS_UNIDENTIFIED`: **False**

`H2_REMAINS=TRUE`; `H3_AUTHORIZED=FALSE`; `DYNAMIC_STATE_SEARCH_AUTHORIZED=FALSE`; `AUTONOMOUS_CONTROLLER_AUTHORIZED=FALSE`. V29 tests: `6 passed in 1.78s`. Full suite: `280 passed, 3 failed, 3 warnings in 52.90s`. Full machine-readable Parquet/JSON/NPZ records, state/token/write/transplant/future hashes and the integrity index accompany this report.
