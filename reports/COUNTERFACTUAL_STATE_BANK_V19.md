# V19 counterfactual state bank

Protocol `0e76a756936e08d71aeee7feffc6dfc8c8e18d572ecd47a29a3048404c843b1a`; split `9d50c11b81f98515fdb2191ec96167bae34e2908585e622084a3160c65503319`; amended intervention `572a827f6fba915490e9262767a1178d1eff7a24b6a17cb55b24f0a69ed529bb`. The original q selection and both corrections remain immutable (`9d4b8f250fcf100cfcef4d9c3e8acfbf1e7f9bda3f6f108510aacf44271637c7`, `0e79c12a9d5aed4dce602d5694994afc46b89e9bbdab8fd5750d0a830093319b`).

Development: 400 train and 100 validation base states, balanced over five task families; disjoint train-only calibration: 20. H2/H4/H8 nested panel: 50/15 train/validation states. Validation reuses V18 clean prompts and is not independent final.

## Frozen q directions

| q | requested channel | direction | α | calibration | reliable fraction | pilot N/R |
|---|---|---|---|---|---|---|
| conv_causal1 | conv | 1 | 1.0 | RELIABLE | 1.0000 | 0.9457 |
| conv_arch24_corrected | conv | 24 | 1.0 | RELIABLE | 1.0000 | 1.7464 |
| joint_causal11 | joint | 11 | 1.0 | RELIABLE | 1.0000 | 0.6945 |
| joint_random2 | joint | 2 | 1.0 | RELIABLE | 1.0000 | 0.0440 |
| rec_random7_weak_control | recurrent | 7 | 1.0 | RELIABLE | 1.0000 | 0.0334 |
| rec_arch4_amended | recurrent | 4 | 1.0 | RELIABLE | 1.0000 | 1.0327 |
| kv_arch14_amended | kv | 14 | 0.5 | RELIABLE | 1.0000 | 0.0228 |
| rec_conv_causal1_amended | rec_conv | 1 | 1.0 | RELIABLE | 1.0000 | 1.3403 |

## Validation BF16 q readback

| q | states | reliable fraction | median raw ‖ΔP‖ | median cosine | median gain | median active-channel survival |
|---|---|---|---|---|---|---|
| conv_arch24_corrected | 100 | 1.0000 | 1583.2938 | 1.0000 | 1.0000 | 0.9995 |
| conv_causal1 | 100 | 1.0000 | 1289.4232 | 1.0000 | 1.0000 | 0.9995 |
| joint_causal11 | 100 | 1.0000 | 624.0678 | 1.0000 | 1.0000 | 0.9961 |
| joint_random2 | 100 | 1.0000 | 22.9323 | 0.9994 | 1.0008 | 0.9199 |
| kv_arch14_amended | 100 | 1.0000 | 5.0878 | 0.9951 | 1.0038 | 0.8562 |
| rec_arch4_amended | 100 | 1.0000 | 108.0083 | 1.0000 | 1.0000 | 0.9994 |
| rec_conv_causal1_amended | 100 | 1.0000 | 1290.6626 | 1.0000 | 1.0000 | 0.9991 |
| rec_random7_weak_control | 100 | 1.0000 | 1.6690 | 0.9997 | 1.0002 | 0.9675 |

Teacher manifest `2143b1e795aa932aeea2ea94a09f5b368ca8449fcf5c89afb10b8604096e7aa3` covers 500 states; 50 new h8 clean-greedy sequences were frozen before factorial responses and have exact V18 h1 prefixes. Every four-way branch uses the same frozen sequence per state.

Boundary-held current J is read once before q; all 13120 validation factorial rows have zero boundary J difference and identical J hashes. P0/Pq persistent snapshot hashes differ on every row. Pq is constructed once per q and cloned for its no-action and action trajectories; P0 is handled analogously. This is an active boundary-held-J counterfactual, not natural same-J matching.

Validation four-way rows: 13120, train rows: 48000. Factorial Parquet records contain Y00, Y01, Y10, Y11 normalized 288D endpoints, realized q/action readback, exact hashes and status denominators.

h16 was not run under the frozen compute-priority rule; h1/h2/h4/h8 are complete on their declared panels.
