# Oracle low-rank causal state — V11

The bases are fitted on the training split only. REC, conv, and KV coordinates are exact
block-normalized dual-PCA coordinates of raw teacher deltas; retained coordinates are
decoded by linear combinations of centered training raw deltas. No learned neural decoder
is used.

| method/condition | h | direction | magnitude | semantic | output | sign | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| oracle_factorized_pca_d128 | 1 | 0.182 | 0.332 | 0.048 | 0.225 | 0.540 | FAIL |
| oracle_factorized_pca_d128 | 2 | 0.300 | 0.564 | 0.044 | 0.328 | 0.640 | FAIL |
| oracle_factorized_pca_d128 | 4 | 0.389 | 0.775 | 0.062 | 0.384 | 0.609 | FAIL |
| oracle_factorized_pca_d128 | 8 | 0.483 | 0.975 | 0.086 | 0.459 | 0.720 | FAIL |
| oracle_factorized_pca_d128 | 16 | 0.465 | 0.940 | 0.066 | 0.478 | 0.596 | FAIL |
| oracle_factorized_pca_d192 | 1 | 0.184 | 0.328 | 0.040 | 0.221 | 0.620 | FAIL |
| oracle_factorized_pca_d192 | 2 | 0.312 | 0.567 | 0.040 | 0.308 | 0.680 | FAIL |
| oracle_factorized_pca_d192 | 4 | 0.400 | 0.770 | 0.062 | 0.419 | 0.761 | FAIL |
| oracle_factorized_pca_d192 | 8 | 0.480 | 0.976 | 0.074 | 0.453 | 0.760 | FAIL |
| oracle_factorized_pca_d192 | 16 | 0.468 | 0.947 | 0.086 | 0.438 | 0.681 | FAIL |
| oracle_factorized_pca_d256 | 1 | 0.173 | 0.342 | 0.064 | 0.237 | 0.560 | FAIL |
| oracle_factorized_pca_d256 | 2 | 0.308 | 0.571 | 0.078 | 0.355 | 0.540 | FAIL |
| oracle_factorized_pca_d256 | 4 | 0.407 | 0.780 | 0.072 | 0.376 | 0.543 | FAIL |
| oracle_factorized_pca_d256 | 8 | 0.487 | 0.979 | 0.090 | 0.485 | 0.760 | FAIL |
| oracle_factorized_pca_d256 | 16 | 0.461 | 0.947 | 0.070 | 0.444 | 0.553 | FAIL |
| oracle_factorized_pca_d320 | 1 | 0.211 | 0.342 | 0.054 | 0.259 | 0.500 | FAIL |
| oracle_factorized_pca_d320 | 2 | 0.315 | 0.574 | 0.032 | 0.352 | 0.640 | FAIL |
| oracle_factorized_pca_d320 | 4 | 0.401 | 0.773 | 0.058 | 0.409 | 0.457 | FAIL |
| oracle_factorized_pca_d320 | 8 | 0.469 | 0.972 | 0.062 | 0.501 | 0.780 | FAIL |
| oracle_factorized_pca_d320 | 16 | 0.465 | 0.939 | 0.090 | 0.493 | 0.660 | FAIL |
| oracle_factorized_pca_d384 | 1 | 0.227 | 0.345 | 0.082 | 0.259 | 0.580 | FAIL |
| oracle_factorized_pca_d384 | 2 | 0.331 | 0.578 | 0.058 | 0.348 | 0.700 | FAIL |
| oracle_factorized_pca_d384 | 4 | 0.414 | 0.787 | 0.054 | 0.469 | 0.609 | FAIL |
| oracle_factorized_pca_d384 | 8 | 0.481 | 0.983 | 0.074 | 0.510 | 0.700 | FAIL |
| oracle_factorized_pca_d384 | 16 | 0.464 | 0.945 | 0.112 | 0.479 | 0.638 | FAIL |
| oracle_factorized_pca_d448 | 1 | 0.246 | 0.366 | 0.074 | 0.276 | 0.620 | FAIL |
| oracle_factorized_pca_d448 | 2 | 0.360 | 0.608 | 0.068 | 0.389 | 0.620 | FAIL |
| oracle_factorized_pca_d448 | 4 | 0.438 | 0.798 | 0.078 | 0.403 | 0.587 | FAIL |
| oracle_factorized_pca_d448 | 8 | 0.483 | 0.967 | 0.074 | 0.471 | 0.600 | FAIL |
| oracle_factorized_pca_d448 | 16 | 0.482 | 0.932 | 0.096 | 0.472 | 0.574 | FAIL |
| oracle_factorized_pca_d512 | 1 | 0.346 | 0.409 | 0.102 | 0.293 | 0.580 | FAIL |
| oracle_factorized_pca_d512 | 2 | 0.420 | 0.644 | 0.082 | 0.407 | 0.500 | FAIL |
| oracle_factorized_pca_d512 | 4 | 0.466 | 0.819 | 0.068 | 0.445 | 0.565 | FAIL |
| oracle_factorized_pca_d512 | 8 | 0.486 | 0.979 | 0.094 | 0.495 | 0.740 | FAIL |
| oracle_factorized_pca_d512 | 16 | 0.485 | 0.957 | 0.078 | 0.478 | 0.596 | FAIL |
| oracle_factorized_pca_d64 | 1 | 0.172 | 0.332 | 0.070 | 0.251 | 0.520 | FAIL |
| oracle_factorized_pca_d64 | 2 | 0.307 | 0.566 | 0.046 | 0.295 | 0.620 | FAIL |
| oracle_factorized_pca_d64 | 4 | 0.392 | 0.786 | 0.056 | 0.416 | 0.674 | FAIL |
| oracle_factorized_pca_d64 | 8 | 0.488 | 0.973 | 0.100 | 0.490 | 0.720 | FAIL |
| oracle_factorized_pca_d64 | 16 | 0.466 | 0.939 | 0.082 | 0.468 | 0.574 | FAIL |
| oracle_joint_pca_d128 | 1 | 0.215 | 0.337 | 0.076 | 0.268 | 0.640 | FAIL |
| oracle_joint_pca_d128 | 2 | 0.321 | 0.569 | 0.030 | 0.339 | 0.700 | FAIL |
| oracle_joint_pca_d128 | 4 | 0.407 | 0.778 | 0.060 | 0.421 | 0.674 | FAIL |
| oracle_joint_pca_d128 | 8 | 0.475 | 0.982 | 0.092 | 0.492 | 0.800 | FAIL |
| oracle_joint_pca_d128 | 16 | 0.475 | 0.938 | 0.068 | 0.463 | 0.702 | FAIL |
| oracle_joint_pca_d192 | 1 | 0.303 | 0.353 | 0.102 | 0.291 | 0.620 | FAIL |
| oracle_joint_pca_d192 | 2 | 0.368 | 0.584 | 0.040 | 0.376 | 0.580 | FAIL |
| oracle_joint_pca_d192 | 4 | 0.424 | 0.783 | 0.058 | 0.433 | 0.587 | FAIL |
| oracle_joint_pca_d192 | 8 | 0.474 | 0.972 | 0.092 | 0.486 | 0.640 | FAIL |
| oracle_joint_pca_d192 | 16 | 0.476 | 0.947 | 0.084 | 0.452 | 0.638 | FAIL |
| oracle_joint_pca_d256 | 1 | 0.410 | 0.388 | 0.170 | 0.427 | 0.580 | FAIL |
| oracle_joint_pca_d256 | 2 | 0.422 | 0.607 | 0.100 | 0.440 | 0.620 | FAIL |
| oracle_joint_pca_d256 | 4 | 0.448 | 0.793 | 0.076 | 0.478 | 0.630 | FAIL |
| oracle_joint_pca_d256 | 8 | 0.485 | 0.973 | 0.084 | 0.495 | 0.660 | FAIL |
| oracle_joint_pca_d256 | 16 | 0.481 | 0.944 | 0.086 | 0.472 | 0.723 | FAIL |
| oracle_joint_pca_d320 | 1 | 0.717 | 0.621 | 0.362 | 0.662 | 0.700 | FAIL |
| oracle_joint_pca_d320 | 2 | 0.617 | 0.725 | 0.178 | 0.559 | 0.700 | FAIL |
| oracle_joint_pca_d320 | 4 | 0.545 | 0.855 | 0.116 | 0.503 | 0.739 | FAIL |
| oracle_joint_pca_d320 | 8 | 0.497 | 0.985 | 0.102 | 0.493 | 0.660 | FAIL |
| oracle_joint_pca_d320 | 16 | 0.502 | 0.968 | 0.092 | 0.457 | 0.660 | FAIL |
| oracle_joint_pca_d384 | 1 | 0.872 | 0.849 | 0.536 | 0.831 | 0.800 | FAIL |
| oracle_joint_pca_d384 | 2 | 0.768 | 0.872 | 0.300 | 0.736 | 0.720 | FAIL |
| oracle_joint_pca_d384 | 4 | 0.645 | 0.938 | 0.186 | 0.607 | 0.804 | FAIL |
| oracle_joint_pca_d384 | 8 | 0.513 | 0.993 | 0.118 | 0.508 | 0.640 | FAIL |
| oracle_joint_pca_d384 | 16 | 0.513 | 0.984 | 0.092 | 0.503 | 0.596 | FAIL |
| oracle_joint_pca_d448 | 1 | 0.915 | 0.939 | 0.636 | 0.890 | 0.820 | FAIL |
| oracle_joint_pca_d448 | 2 | 0.811 | 0.951 | 0.332 | 0.771 | 0.700 | FAIL |
| oracle_joint_pca_d448 | 4 | 0.674 | 0.961 | 0.184 | 0.652 | 0.761 | FAIL |
| oracle_joint_pca_d448 | 8 | 0.523 | 1.010 | 0.098 | 0.510 | 0.660 | FAIL |
| oracle_joint_pca_d448 | 16 | 0.532 | 0.984 | 0.114 | 0.518 | 0.702 | FAIL |
| oracle_joint_pca_d512 | 1 | 0.921 | 0.963 | 0.618 | 0.890 | 0.820 | FAIL |
| oracle_joint_pca_d512 | 2 | 0.816 | 0.964 | 0.326 | 0.779 | 0.700 | FAIL |
| oracle_joint_pca_d512 | 4 | 0.684 | 0.973 | 0.240 | 0.642 | 0.739 | FAIL |
| oracle_joint_pca_d512 | 8 | 0.523 | 0.997 | 0.118 | 0.514 | 0.720 | FAIL |
| oracle_joint_pca_d512 | 16 | 0.534 | 0.994 | 0.134 | 0.494 | 0.660 | FAIL |
| oracle_joint_pca_d64 | 1 | 0.177 | 0.327 | 0.088 | 0.243 | 0.460 | FAIL |
| oracle_joint_pca_d64 | 2 | 0.309 | 0.574 | 0.056 | 0.346 | 0.440 | FAIL |
| oracle_joint_pca_d64 | 4 | 0.396 | 0.772 | 0.062 | 0.392 | 0.717 | FAIL |
| oracle_joint_pca_d64 | 8 | 0.479 | 0.975 | 0.092 | 0.481 | 0.640 | FAIL |
| oracle_joint_pca_d64 | 16 | 0.484 | 0.949 | 0.066 | 0.476 | 0.702 | FAIL |

Long-horizon authorized methods: `[]`.
Decision-tree outcome: **C** —
tested low-rank oracle projections do not establish writable sufficiency.

Machine records: `results/v11/processed/oracle_lowrank_causal_v11.parquet` (`d4434d624419e318a6c8d06252f9bdd31bafbcf7aa9e236d5a6876a50e671e93`).
