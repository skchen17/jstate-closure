
# V22 Complete Report

## Identity

- Name: **Causal Action Manifold Expansion and Input-Side Operator Geometry**
- Parent: `351da6f061c3ed13e90324f50afc30f53a8796de`
- Protocol hash: `fc2efccc8fa1f6072a0dbcc3a810c107bb09e6eedd68813b4fca6e800d1d85f5`
- Candidate pool hash: `658f49a2d073a6c7f00624817cabd03e62ed8083d1d701e1d4d5b9bcb10293dd`
- Train / validation / independent-final hashes: `a5377e3897dd295c0267814c35f76844de4ca8331b25f56dd0d93cdd73dd1590` / `793cab02928568a7201c863993943c76da566809bb51fb6ce7f573e91fa14d5f` / `dc340edd9a827d4c4f433a2cd0102d64fe0c96aceb875895d3bc47c2b7f46eb4`
- Reliable candidates: `480 / 512`

## Outcome

Formal outcome: **V22-D_ACTION_DATA_LIMITED**.

Metric-corrected input ranks are JVP `[11.0, 15.0, 29.0]` and finite `[14.0, 20.0, 34.0]`. Both input and output subspaces rotate under P0→Pq. The best practical coordinate is `Z6` dimension `4`, but its unseen-direction L2 is `0.9715` and the even/odd negative L2 is `1.0065`. The practical gate therefore fails.

Expanded action data did not establish stable saturation across all frozen strategies. V22 therefore does not reinterpret the remaining error as operator nonexistence. The compact operator stage and both final-response openings remain closed.

## Geometry

# Input-Side Causal Geometry — V22

The 63 reliable V21 directions were corrected with the full 14,397-D raw-action Gram before decomposition. The Gram rank is 63, retained condition number is 5.99e+05, and the relative eigenvalue floor is `1e-8`.

| operator | input r90/r95/r99 | input P0→Pq angle | output P0→Pq angle |
|---|---:|---:|---:|
| exact JVP | [11.0, 15.0, 29.0] | 18.5340° | 18.2923° |
| finite | [14.0, 20.0, 34.0] | 31.4067° | 35.3716° |

JVP/finite input-subspace overlap is 0.5334 at P0 and 0.4576 at Pq. Input V rotates materially, so a fixed global action coordinate is structurally incomplete. Output rank is not used as a claim that the raw action space has the same dimension.


## Action pool and experiment design

# Action Pool Calibration — V22

The candidate pool contains 512 actions: 506 non-final V13 directions plus six frozen balanced combinations. Historical final-six geometry and responses remained sealed. Five train-only calibration states measured both signs at amplitudes 0.25, 0.5, and 1.0.

Symmetric reliable candidates: **480 / 512**. Four initially frozen validation candidates failed symmetric reliability and were replaced, before any expanded response measurement, by the recorded append-only reserve amendment.


# Action Experiment Design — V22

Strategies: `random_architecture_balanced, raw_geometry_maximin, causal_d_optimal`. Nested counts: `[12, 24, 48, 96, 128]`. A common 128-direction measured panel permits a controlled ranking comparison; 32 reliable validation directions and 32 independent-final directions were frozen separately. The causal D-optimal design used development-state exact JVP only and is an offline experiment-design method, not an inference-time oracle.

Selection hashes: `{"causal_d_optimal": "44a4da39a4d950e0018dd6e95a3339f9c1afe6f5b2bfd9e96d08045f4fa706fd", "random_architecture_balanced": "b2742ecbbff548c434870392af11fb6aad7d10c6efa042af3259282521f83762", "raw_geometry_maximin": "9c27e515e95cf1d48fd07c2c1e27e13eaae23411a2c24b195f3546beae806df3"}`.


## Scaling and coordinates

# Action Data Scaling — V22

| strategy | actions | S1 cross-state L2 | local S2 action-ceiling L2 | local cosine |
|---|---:|---:|---:|---:|
| causal_d_optimal | 12 | 1.0016 | 0.9951 | 0.0558 |
| causal_d_optimal | 24 | 1.0123 | 0.9954 | 0.2747 |
| causal_d_optimal | 48 | 1.0310 | 0.9958 | 0.4576 |
| causal_d_optimal | 96 | 1.4368 | 1.4571 | 0.6195 |
| causal_d_optimal | 128 | 1.3226 | 1.5474 | 0.6206 |
| random_architecture_balanced | 12 | 1.1079 | 1.0184 | 0.5871 |
| random_architecture_balanced | 24 | 1.1434 | 1.0089 | 0.6077 |
| random_architecture_balanced | 48 | 1.1511 | 1.0029 | 0.6191 |
| random_architecture_balanced | 96 | 1.0393 | 1.0170 | 0.5951 |
| random_architecture_balanced | 128 | 1.0366 | 1.0074 | 0.5945 |
| raw_geometry_maximin | 12 | 1.0089 | 0.9272 | 0.5606 |
| raw_geometry_maximin | 24 | 1.0075 | 0.9236 | 0.5719 |
| raw_geometry_maximin | 48 | 1.0080 | 0.9209 | 0.5738 |
| raw_geometry_maximin | 96 | 1.0107 | 0.8921 | 0.5851 |
| raw_geometry_maximin | 128 | 1.0366 | 1.0074 | 0.5945 |

Saturation by strategy: `{"causal_d_optimal": false, "random_architecture_balanced": true, "raw_geometry_maximin": false}`. Overall saturation is not established; `ACTION_DATA_LIMITED=TRUE`. The adverse 128-action point is reported, not hidden.


# Causal Action Coordinates — V22

| coordinate | dim | unseen direction L2 | direction+sign L2 | local S2 ceiling L2 |
|---|---:|---:|---:|---:|
| Z1 | None | 1.0366 | 1.0485 | 1.0074 |
| Z2 | None | 2.0060 | 2.0167 | 1.3435 |
| Z6 | 4 | 0.9715 | 1.0055 | 0.8954 |
| Z7 | 4 | 1.6662 | 1.6804 | 2.0703 |
| Z6 | 8 | 1.9637 | 2.0433 | 1.3481 |
| Z7 | 8 | 16.4925 | 16.5306 | 14.8243 |
| Z6 | 16 | 1.1336 | 1.1681 | 1.0670 |
| Z7 | 16 | 8.3928 | 8.3971 | 8.8602 |
| Z6 | 32 | 1.1096 | 1.1272 | 1.0717 |
| Z7 | 32 | 4.4082 | 4.4196 | 5.9452 |
| Z6 | 64 | 1.0778 | 1.0994 | 1.0641 |
| Z7 | 64 | 3.8684 | 3.8862 | 5.6224 |
| Z6 | 128 | 1.0778 | 1.0994 | 1.0641 |
| Z7 | 128 | 3.8684 | 3.8862 | 5.6224 |
| Z8_poly2 | 64 | 2.1006 | 2.1110 | 1.3564 |
| Z8_poly3 | 64 | 4.9977 | 4.9925 | 2.3846 |
| Z8_rff | 64 | 0.9992 | 1.0202 | 0.8896 |

Best practical candidate: `Z6` dimension `4`, unseen-direction L2 `0.9715`. It does not pass the 0.30 gate.


## Nonlinearity and composition

# Even/Odd Action Geometry — V22

Explicit decomposition yields reconstructed positive L2 `0.9715` and negative L2 `1.0065`. The odd-symmetry negative baseline is `1.0055`; the explicit split does not materially repair sign generalization.


# Action Scale Law — V22

Amplitudes 0.25, 0.5, and 1.0 were measured on a separately frozen panel. Positive/negative monotone-gain validation L2 is `0.1979` / `0.2534`. Linear scale L2 is `0.3778` / `0.5402`. Scale-only nonlinearity is identifiable, but this does not solve direction generalization.


# Action Composition — V22

Additive finite superposition L2: positive pair `0.1369`, negative pair `0.1609`, positive dense `0.3110`, negative dense `0.4173`. Pair composition passes the 0.30 diagnostic; dense and negative dense do not both pass.


## State-conditioned and low-rank tests

# State-Conditioned Action Chart — V22

V22-G2 authorized this test. Fixed-global odd-response L2 is `0.9983`; the low-rank S1-conditioned local chart is `1.0016`. Gain is `-0.0033`; therefore state-conditioned transport is structurally motivated but not empirically beneficial in the tested constrained model.


# Low-Rank Operator Identification — V22

| rank | unseen-direction L2 | cosine |
|---:|---:|---:|
| 4 | 1.0365 | 0.2275 |
| 8 | 1.0068 | 0.2761 |
| 16 | 1.0016 | 0.3040 |
| 32 | 1.0016 | 0.3040 |

No tested rank passes. A compact operator-state coordinate is not identified, and no nonexistence claim is made while action scaling is unsaturated.


## Final authorization flags

- `V22_G1_STABLE_INPUT_CAUSAL_SUBSPACE = FALSE`
- `V22_G2_STATE_DEPENDENT_INPUT_CAUSAL_SUBSPACE = TRUE`
- `V22_G3_DIFFERENTIAL_FINITE_INPUT_DIVERGENCE = TRUE`
- `PRACTICAL_ACTION_COORDINATE_PASS = FALSE`
- `ACTION_COVERAGE_BOTTLENECK_RESOLVED = FALSE`
- `ACTION_DATA_LIMITED = TRUE`
- `CROSS_ACTION_OPERATOR_STILL_UNIDENTIFIED_AFTER_COVERAGE = FALSE`
- `SIGN_NONLINEARITY_IS_PRIMARY_LIMIT = FALSE`
- `COMPACT_OPERATOR_SEARCH_REOPENED = FALSE`
- `RAW_TO_OPERATOR_ENCODER_AUTHORIZED = FALSE`
- `H2_REMAINS = TRUE`
- `H3_AUTHORIZED = FALSE`
- `DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`
- `HISTORICAL_FINAL_SIX_OPENED = FALSE`
- `NEW_INDEPENDENT_FINAL_OPENED = FALSE`

## Verification

V22 tests: `5 passed, 0 failed`. Full suite: `238 passed, 2 failed`; both failures are inherited cumulative FINAL_REPORT hash assertions from V14/V16 and were not hidden by rewriting historical manifests.

## Scientific answers


1. Input r90/r95/r99: JVP `[11.0, 15.0, 29.0]`; finite `[14.0, 20.0, 34.0]`.
2. Input rank is lower than the 63-direction metric domain but materially larger than the old output-only 5–7 estimate; it is not evidence that raw action space is 7-D.
3. Output U rotates: JVP `18.2923°`, finite `35.3716°`.
4. Input V rotates: JVP `18.5340°`, finite `31.4067°`.
5. JVP/finite input overlap: P0 `0.5334`, Pq `0.4576`; divergence is material at Pq.
6. A fixed global action coordinate is not supported.
7. A state-conditioned chart is structurally indicated, but the tested chart did not improve validation.
8. Reliable directions: `480` of 512.
9. Full 12/24/48/96/128 curves are in ACTION_DATA_SCALING_V22.md.
10. Overall saturation: `FALSE`.
11. Raw maximin gives the best measured local action ceiling; causal D-optimal does not.
12. Causal D-optimal does not beat random/maximin consistently.
13. Z6 beats raw Z1 in the best cross-state comparison but remains far above 0.30.
14. Z7 does not solve unseen action prediction.
15. Polynomial/RBF kernels do not pass.
16. Explicit even/odd does not materially fix unseen sign.
17. Monotone scale modeling improves scale-only L2 to `0.1979` / `0.2534`.
18. No tested coverage metric strongly predicts held-out error.
19. State-conditioned transport changes L2 by `-0.0033` and does not help.
20. No compact operator coordinate emerges.
21. `k_operator_min = NONE`.
22. Historical final six remain sealed.
23. New independent final remains frozen and unopened because no finalist qualified.
24. `RAW_TO_OPERATOR_ENCODER_AUTHORIZED = FALSE`.
25. H2 remains.
26. H3 is not authorized.
27. Dynamic-state search is not authorized.

