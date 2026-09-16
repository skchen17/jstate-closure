# Local causal oracle — V12

Global PCA、global causal covariance basis、current-J-neighborhood local PCA 与 local causal basis 都通过同一 raw-state dual reconstruction interface 比较。Local 512D 因 512-neighbor centered rank≤511 被正式标记 `NOT_IDENTIFIED_RANK_LIMIT`。

## Development h1

| method | h | direction | magnitude | semantic | output | sign | all-family gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| global_architecture_pca_d128 | 1 | 0.206 | 0.337 | 0.062 | 0.252 | 0.520 | FAIL |
| global_architecture_pca_d256 | 1 | 0.431 | 0.388 | 0.170 | 0.420 | 0.640 | FAIL |
| global_architecture_pca_d384 | 1 | 0.870 | 0.851 | 0.534 | 0.833 | 0.740 | FAIL |
| global_architecture_pca_d512 | 1 | 0.923 | 0.968 | 0.620 | 0.897 | 0.860 | FAIL |
| global_architecture_pca_d64 | 1 | 0.189 | 0.326 | 0.066 | 0.225 | 0.620 | FAIL |
| global_causal_basis_d128 | 1 | 0.902 | 0.944 | 0.598 | 0.865 | 0.780 | FAIL |
| global_causal_basis_d256 | 1 | 0.913 | 0.963 | 0.652 | 0.875 | 0.760 | FAIL |
| global_causal_basis_d384 | 1 | 0.918 | 0.960 | 0.620 | 0.886 | 0.780 | FAIL |
| global_causal_basis_d512 | 1 | 0.922 | 0.968 | 0.628 | 0.896 | 0.820 | FAIL |
| global_causal_basis_d64 | 1 | 0.891 | 0.940 | 0.570 | 0.861 | 0.800 | FAIL |
| global_combined_pca_d128 | 1 | 0.188 | 0.336 | 0.084 | 0.228 | 0.520 | FAIL |
| global_combined_pca_d256 | 1 | 0.477 | 0.472 | 0.210 | 0.394 | 0.660 | FAIL |
| global_combined_pca_d384 | 1 | 0.766 | 0.725 | 0.432 | 0.698 | 0.760 | FAIL |
| global_combined_pca_d512 | 1 | 0.924 | 0.977 | 0.614 | 0.894 | 0.800 | FAIL |
| global_combined_pca_d64 | 1 | 0.176 | 0.332 | 0.060 | 0.204 | 0.460 | FAIL |
| local_causal_basis_d128 | 1 | 0.904 | 0.958 | 0.610 | 0.863 | 0.840 | FAIL |
| local_causal_basis_d256 | 1 | 0.915 | 0.974 | 0.618 | 0.885 | 0.760 | FAIL |
| local_causal_basis_d384 | 1 | 0.920 | 0.966 | 0.634 | 0.888 | 0.840 | FAIL |
| local_causal_basis_d64 | 1 | 0.891 | 0.960 | 0.572 | 0.859 | 0.880 | FAIL |
| local_pca_d128 | 1 | 0.253 | 0.382 | 0.112 | 0.252 | 0.560 | FAIL |
| local_pca_d256 | 1 | 0.502 | 0.569 | 0.184 | 0.522 | 0.680 | FAIL |
| local_pca_d384 | 1 | 0.832 | 0.839 | 0.444 | 0.809 | 0.800 | FAIL |
| local_pca_d64 | 1 | 0.207 | 0.350 | 0.082 | 0.255 | 0.580 | FAIL |

## Development staged h2/h4/h8 (h16 only after h1 pass)

| method | h | direction | magnitude | semantic | output | sign | all-family gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| global_architecture_pca_d512 | 2 | 0.816 | 0.970 | 0.350 | 0.778 | 0.760 | FAIL |
| global_architecture_pca_d512 | 4 | 0.681 | 0.981 | 0.228 | 0.640 | 0.674 | FAIL |
| global_architecture_pca_d512 | 8 | 0.521 | 0.999 | 0.104 | 0.508 | 0.720 | FAIL |
| global_causal_basis_d512 | 2 | 0.808 | 0.968 | 0.352 | 0.790 | 0.740 | FAIL |
| global_causal_basis_d512 | 4 | 0.671 | 0.974 | 0.204 | 0.644 | 0.674 | FAIL |
| global_causal_basis_d512 | 8 | 0.532 | 1.009 | 0.104 | 0.518 | 0.580 | FAIL |
| local_causal_basis_d384 | 2 | 0.818 | 0.961 | 0.336 | 0.788 | 0.760 | FAIL |
| local_causal_basis_d384 | 4 | 0.688 | 0.976 | 0.222 | 0.623 | 0.717 | FAIL |
| local_causal_basis_d384 | 8 | 0.516 | 1.002 | 0.096 | 0.488 | 0.640 | FAIL |
| local_pca_d384 | 2 | 0.734 | 0.879 | 0.254 | 0.701 | 0.620 | FAIL |
| local_pca_d384 | 4 | 0.645 | 0.948 | 0.160 | 0.618 | 0.739 | FAIL |
| local_pca_d384 | 8 | 0.520 | 1.010 | 0.088 | 0.522 | 0.680 | FAIL |

## New independent V12 confirmation

| method | h | direction | magnitude | semantic | output | sign | all-family gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| global_architecture_pca_d512 | 1 | 0.918 | 0.958 | 0.644 | 0.896 | 0.840 | FAIL |
| global_architecture_pca_d512 | 2 | 0.806 | 0.972 | 0.342 | 0.772 | 0.820 | FAIL |
| global_architecture_pca_d512 | 4 | 0.677 | 1.000 | 0.192 | 0.638 | 0.600 | FAIL |
| global_architecture_pca_d512 | 8 | 0.531 | 1.001 | 0.106 | 0.519 | 0.776 | FAIL |
| global_causal_basis_d512 | 1 | 0.920 | 0.964 | 0.624 | 0.893 | 0.820 | FAIL |
| global_causal_basis_d512 | 2 | 0.811 | 0.974 | 0.356 | 0.775 | 0.780 | FAIL |
| global_causal_basis_d512 | 4 | 0.663 | 1.003 | 0.210 | 0.618 | 0.689 | FAIL |
| global_causal_basis_d512 | 8 | 0.531 | 1.005 | 0.098 | 0.521 | 0.816 | FAIL |
| local_causal_basis_d384 | 1 | 0.915 | 0.963 | 0.634 | 0.885 | 0.820 | FAIL |
| local_causal_basis_d384 | 2 | 0.808 | 0.978 | 0.340 | 0.777 | 0.800 | FAIL |
| local_causal_basis_d384 | 4 | 0.672 | 1.004 | 0.202 | 0.643 | 0.756 | FAIL |
| local_causal_basis_d384 | 8 | 0.528 | 0.993 | 0.122 | 0.528 | 0.796 | FAIL |
| local_pca_d384 | 1 | 0.829 | 0.846 | 0.486 | 0.818 | 0.800 | FAIL |
| local_pca_d384 | 2 | 0.719 | 0.908 | 0.276 | 0.702 | 0.700 | FAIL |
| local_pca_d384 | 4 | 0.628 | 0.982 | 0.174 | 0.610 | 0.822 | FAIL |
| local_pca_d384 | 8 | 0.515 | 0.996 | 0.094 | 0.502 | 0.755 | FAIL |

- Confirmed authorized methods: `[]`
- Smallest causally validated dimension: `None`
- Formal outcome: **V12-A — MORE_DATA_REQUIRED**
