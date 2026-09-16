# Causal-state manifold audit — V11

All reference geometry is fitted exclusively to training teacher raw-intervention deltas.
Distances are reported in exact architecture-channel score coordinates.

| state | channel | NN dist | Mahalanobis | local PCA err | outside-128 | cycle err |
|---|---|---:|---:|---:|---:|---:|
| clean_zero | conv | 0.980 | 0.000 | 1.000 | 0.627 | 1.227e+08 |
| clean_zero | kv | 0.999 | 0.000 | 1.000 | 0.710 | 2.015e+08 |
| clean_zero | recurrent | 0.990 | 0.000 | 1.000 | 0.647 | 1.194e+08 |
| factorized_multistep_direction_int_d512 | conv | 0.969 | 9.954 | 0.977 | 0.987 | 1.784e-07 |
| factorized_multistep_direction_int_d512 | kv | 0.964 | 8.354 | 0.979 | 0.984 | 2.138e-07 |
| factorized_multistep_direction_int_d512 | recurrent | 0.971 | 13.518 | 0.958 | 0.990 | 2.277e-07 |
| oracle_factorized_pca_d512 | conv | 0.998 | 7.546 | 0.998 | 0.219 | 3.973e-07 |
| oracle_factorized_pca_d512 | kv | 0.981 | 5.592 | 0.995 | 0.000 | 3.400e-07 |
| oracle_factorized_pca_d512 | recurrent | 1.027 | 12.381 | 0.990 | 0.536 | 2.859e-07 |
| oracle_joint_pca_d512 | conv | 0.921 | 14.103 | 0.937 | 0.828 | 1.897e-07 |
| oracle_joint_pca_d512 | kv | 0.921 | 14.103 | 0.934 | 0.839 | 2.399e-07 |
| oracle_joint_pca_d512 | recurrent | 0.921 | 14.103 | 0.930 | 0.846 | 2.153e-07 |
| teacher_raw | conv | 0.984 | 16.733 | 0.957 | 0.836 | 2.090e-07 |
| teacher_raw | kv | 0.908 | 16.748 | 0.888 | 0.828 | 2.422e-07 |
| teacher_raw | recurrent | 1.037 | 20.185 | 0.948 | 0.822 | 2.276e-07 |
| unified_causal_d512 | conv | 2.263 | 52.940 | 0.998 | 0.884 | 3.298e-07 |
| unified_causal_d512 | kv | 1.361 | 25.918 | 0.998 | 0.860 | 2.372e-07 |
| unified_causal_d512 | recurrent | 1.598 | 33.705 | 0.999 | 0.868 | 2.578e-07 |

## Joint channel compatibility

- `clean_zero` pairwise nearest-neighbor agreement: 0.000; all-three: 0.000.
- `factorized_multistep_direction_int_d512` pairwise nearest-neighbor agreement: 0.460; all-three: 0.320.
- `oracle_factorized_pca_d512` pairwise nearest-neighbor agreement: 0.213; all-three: 0.080.
- `oracle_joint_pca_d512` pairwise nearest-neighbor agreement: 0.987; all-three: 0.980.
- `teacher_raw` pairwise nearest-neighbor agreement: 0.533; all-three: 0.420.
- `unified_causal_d512` pairwise nearest-neighbor agreement: 0.000; all-three: 0.000.

Machine records: `results/v11/processed/causal_state_manifold_v11.parquet` (`06f13b242adae1b38ac10e7e76c2401b5b20850fc13e9161a7e60efbda93465e`).


<!-- V11_MANIFOLD_AMENDMENT_1 -->
## Manifold metric amendment 1

The original clean-zero *relative* cycle error divided by `||query||=0` and is undefined.
The original record remains preserved; corrected records report it as null and report the
absolute score-space cycle error separately. All nonzero teacher/decoded relative cycle
metrics are unchanged.

| channel | relative cycle error | absolute cycle error |
|---|---:|---:|
| conv | undefined | 1.227e-12 |
| kv | undefined | 2.015e-12 |
| recurrent | undefined | 1.194e-12 |

Amendment freeze: `f17f62532f1136c9cc73109ecf55e2f6689b55638313979eb9c1d5068ac5c14f`. Corrected records: `results/v11/processed/causal_state_manifold_v11_amendment_1.parquet` (`64e60138414fe876f5140586fd6b72058d8f9c05e76b03ead499720da078017b`).
