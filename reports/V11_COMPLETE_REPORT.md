# V11 complete report

> This is the canonical single-file bundle for this version. It combines the 
> version-specific FINAL_REPORT section, every standalone version report, and 
> an integrity index of the machine-readable records. Standalone reports remain 
> preserved for direct navigation.

## Bundle provenance

- Source commit: `5e4025a6e32a70e99ec0c44a3394d0028b70dcd2`
- Generated at: `2026-09-16T15:23:57.036714+00:00`
- Included standalone reports: `6`
- Indexed machine-record files: `26`
- Generator: `scripts/build_complete_version_report.py`

## Version summary and adjudication

## V11 — Architecture-resolved causal geometry

V11 used a new 50-pair independent confirmatory bank (10/family), disjoint from the 50 v10
development pairs. Development split hash:
`882c53041d9806ddde1915542e132dd14e43db1d146f9de1e53fffd999e4d635`; confirmatory split hash:
`b22941ec58285ad1cbed6f541d9cfe4ab1cee36983faf81dc16986f3f2851df3`; confirmatory program hash:
`db79342cf71638baa9e71b873937ce6931377c65dd314a400bcdf057d343626f`.

Frozen gates: `{"direction_cosine_minimum": 0.8, "magnitude_ratio_maximum": 1.2, "magnitude_ratio_minimum": 0.8, "output_direction_minimum": 0.8, "semantic_delta_agreement_minimum": 0.8, "task_decision_sign_minimum": 0.8}`.
Freeze digests: `{"base": "91cca911c33e1c5829d978fa307a54cfa7093118d039de35785a2fc388986ff3", "confirmatory": "753b48824a27a9b5934b774ba0594dc26d98a4d031c6a8c183b065c55470ca9f", "prepared": "58a31c7022b99b035b915054c64fe1b4e345b886838f75ea4d6ac401fb8675b8", "stage2": "d040b22427c48fcec1b4564b2fb14aa8d742b70047e11c1ce655a707e1308135"}`.

### Channel-wise hybrid interventions

| method/condition | h | direction | magnitude | semantic | output | sign | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| decoded_all | 1 | 0.923 | 0.977 | 0.614 | 0.896 | 0.820 | FAIL |
| decoded_all | 2 | 0.823 | 0.980 | 0.352 | 0.795 | 0.740 | FAIL |
| decoded_all | 4 | 0.688 | 0.987 | 0.206 | 0.660 | 0.696 | FAIL |
| decoded_all | 8 | 0.532 | 0.998 | 0.086 | 0.528 | 0.660 | FAIL |
| decoded_conv | 1 | 0.934 | 0.984 | 0.672 | 0.904 | 0.900 | FAIL |
| decoded_conv | 2 | 0.837 | 0.998 | 0.366 | 0.806 | 0.760 | FAIL |
| decoded_conv | 4 | 0.705 | 0.989 | 0.242 | 0.676 | 0.761 | FAIL |
| decoded_conv | 8 | 0.544 | 1.010 | 0.110 | 0.512 | 0.740 | FAIL |
| decoded_conv_kv | 1 | 0.932 | 0.978 | 0.656 | 0.907 | 0.860 | FAIL |
| decoded_conv_kv | 2 | 0.840 | 0.998 | 0.384 | 0.805 | 0.740 | FAIL |
| decoded_conv_kv | 4 | 0.703 | 0.991 | 0.228 | 0.671 | 0.761 | FAIL |
| decoded_conv_kv | 8 | 0.554 | 1.011 | 0.132 | 0.544 | 0.680 | FAIL |
| decoded_kv | 1 | 0.970 | 0.995 | 0.760 | 0.945 | 0.820 | FAIL |
| decoded_kv | 2 | 0.911 | 0.997 | 0.500 | 0.874 | 0.800 | FAIL |
| decoded_kv | 4 | 0.861 | 0.989 | 0.406 | 0.807 | 0.783 | FAIL |
| decoded_kv | 8 | 0.770 | 1.010 | 0.266 | 0.660 | 0.740 | FAIL |
| decoded_rec | 1 | 0.942 | 0.997 | 0.674 | 0.913 | 0.940 | FAIL |
| decoded_rec | 2 | 0.833 | 0.995 | 0.382 | 0.818 | 0.740 | FAIL |
| decoded_rec | 4 | 0.697 | 0.993 | 0.222 | 0.628 | 0.804 | FAIL |
| decoded_rec | 8 | 0.530 | 1.003 | 0.096 | 0.527 | 0.700 | FAIL |
| decoded_rec_conv | 1 | 0.923 | 0.979 | 0.624 | 0.900 | 0.860 | FAIL |
| decoded_rec_conv | 2 | 0.826 | 0.991 | 0.352 | 0.802 | 0.760 | FAIL |
| decoded_rec_conv | 4 | 0.691 | 0.987 | 0.212 | 0.652 | 0.565 | FAIL |
| decoded_rec_conv | 8 | 0.529 | 1.004 | 0.072 | 0.519 | 0.620 | FAIL |
| decoded_rec_kv | 1 | 0.941 | 1.000 | 0.682 | 0.914 | 0.940 | FAIL |
| decoded_rec_kv | 2 | 0.832 | 0.998 | 0.404 | 0.814 | 0.800 | FAIL |
| decoded_rec_kv | 4 | 0.697 | 0.995 | 0.212 | 0.655 | 0.696 | FAIL |
| decoded_rec_kv | 8 | 0.527 | 1.006 | 0.080 | 0.517 | 0.720 | FAIL |
| teacher_reference | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| teacher_reference | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| teacher_reference | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| teacher_reference | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |

### Oracle low-rank sweep

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

### Unified/factorized and causal-loss ablations

| method/condition | h | direction | magnitude | semantic | output | sign | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| factorized_cache_pca_int_d256 | 1 | 0.179 | 0.331 | 0.074 | 0.236 | 0.620 | FAIL |
| factorized_cache_pca_int_d384 | 1 | 0.207 | 0.344 | 0.072 | 0.210 | 0.440 | FAIL |
| factorized_cache_pca_int_d512 | 1 | 0.289 | 0.384 | 0.084 | 0.314 | 0.600 | FAIL |
| factorized_cache_pca_no_int_d256 | 1 | 0.173 | 0.342 | 0.064 | 0.237 | 0.560 | FAIL |
| factorized_cache_pca_no_int_d384 | 1 | 0.227 | 0.345 | 0.082 | 0.259 | 0.580 | FAIL |
| factorized_cache_pca_no_int_d512 | 1 | 0.346 | 0.409 | 0.102 | 0.293 | 0.580 | FAIL |
| factorized_causal_composite_int_d256 | 1 | 0.745 | 0.744 | 0.386 | 0.679 | 0.680 | FAIL |
| factorized_causal_composite_int_d384 | 1 | 0.762 | 0.752 | 0.396 | 0.692 | 0.680 | FAIL |
| factorized_causal_composite_int_d512 | 1 | 0.816 | 0.818 | 0.432 | 0.770 | 0.740 | FAIL |
| factorized_causal_composite_no_int_d256 | 1 | 0.699 | 0.676 | 0.348 | 0.611 | 0.680 | FAIL |
| factorized_causal_composite_no_int_d384 | 1 | 0.743 | 0.730 | 0.376 | 0.709 | 0.640 | FAIL |
| factorized_causal_composite_no_int_d512 | 1 | 0.808 | 0.807 | 0.464 | 0.749 | 0.760 | FAIL |
| factorized_effect_weighted_multistep_int_d256 | 1 | 0.792 | 0.809 | 0.414 | 0.770 | 0.660 | FAIL |
| factorized_effect_weighted_multistep_int_d384 | 1 | 0.833 | 0.825 | 0.494 | 0.789 | 0.720 | FAIL |
| factorized_effect_weighted_multistep_int_d512 | 1 | 0.849 | 0.855 | 0.492 | 0.821 | 0.740 | FAIL |
| factorized_effect_weighted_multistep_no_int_d256 | 1 | 0.774 | 0.749 | 0.416 | 0.721 | 0.700 | FAIL |
| factorized_effect_weighted_multistep_no_int_d384 | 1 | 0.815 | 0.784 | 0.446 | 0.771 | 0.680 | FAIL |
| factorized_effect_weighted_multistep_no_int_d512 | 1 | 0.840 | 0.832 | 0.508 | 0.805 | 0.720 | FAIL |
| factorized_h1_direction_int_d256 | 1 | 0.803 | 0.815 | 0.430 | 0.772 | 0.800 | FAIL |
| factorized_h1_direction_int_d384 | 1 | 0.835 | 0.857 | 0.454 | 0.804 | 0.700 | FAIL |
| factorized_h1_direction_int_d512 | 1 | 0.852 | 0.875 | 0.522 | 0.821 | 0.740 | FAIL |
| factorized_h1_direction_no_int_d256 | 1 | 0.760 | 0.730 | 0.376 | 0.713 | 0.740 | FAIL |
| factorized_h1_direction_no_int_d384 | 1 | 0.817 | 0.797 | 0.470 | 0.776 | 0.760 | FAIL |
| factorized_h1_direction_no_int_d512 | 1 | 0.845 | 0.853 | 0.518 | 0.802 | 0.800 | FAIL |
| factorized_manifold_regularized_int_d256 | 1 | 0.761 | 0.750 | 0.374 | 0.707 | 0.740 | FAIL |
| factorized_manifold_regularized_int_d384 | 1 | 0.762 | 0.752 | 0.396 | 0.692 | 0.680 | FAIL |
| factorized_manifold_regularized_int_d512 | 1 | 0.816 | 0.818 | 0.432 | 0.770 | 0.740 | FAIL |
| factorized_manifold_regularized_no_int_d256 | 1 | 0.699 | 0.676 | 0.348 | 0.611 | 0.680 | FAIL |
| factorized_manifold_regularized_no_int_d384 | 1 | 0.751 | 0.725 | 0.366 | 0.664 | 0.700 | FAIL |
| factorized_manifold_regularized_no_int_d512 | 1 | 0.808 | 0.807 | 0.464 | 0.749 | 0.760 | FAIL |
| factorized_multistep_direction_int_d256 | 1 | 0.790 | 0.784 | 0.432 | 0.749 | 0.720 | FAIL |
| factorized_multistep_direction_int_d384 | 1 | 0.836 | 0.841 | 0.472 | 0.790 | 0.720 | FAIL |
| factorized_multistep_direction_int_d512 | 1 | 0.859 | 0.880 | 0.516 | 0.827 | 0.760 | FAIL |
| factorized_multistep_direction_no_int_d256 | 1 | 0.760 | 0.732 | 0.418 | 0.729 | 0.720 | FAIL |
| factorized_multistep_direction_no_int_d384 | 1 | 0.808 | 0.798 | 0.478 | 0.767 | 0.800 | FAIL |
| factorized_multistep_direction_no_int_d512 | 1 | 0.849 | 0.847 | 0.508 | 0.810 | 0.780 | FAIL |
| factorized_output_h1_int_d256 | 1 | 0.657 | 0.651 | 0.318 | 0.601 | 0.660 | FAIL |
| factorized_output_h1_int_d384 | 1 | 0.705 | 0.712 | 0.350 | 0.655 | 0.760 | FAIL |
| factorized_output_h1_int_d512 | 1 | 0.727 | 0.705 | 0.364 | 0.665 | 0.740 | FAIL |
| factorized_output_h1_no_int_d256 | 1 | 0.650 | 0.629 | 0.278 | 0.589 | 0.700 | FAIL |
| factorized_output_h1_no_int_d384 | 1 | 0.710 | 0.685 | 0.356 | 0.650 | 0.700 | FAIL |
| factorized_output_h1_no_int_d512 | 1 | 0.718 | 0.691 | 0.376 | 0.651 | 0.720 | FAIL |
| factorized_semantic_h1_int_d256 | 1 | 0.766 | 0.758 | 0.394 | 0.716 | 0.640 | FAIL |
| factorized_semantic_h1_int_d384 | 1 | 0.756 | 0.757 | 0.392 | 0.688 | 0.640 | FAIL |
| factorized_semantic_h1_int_d512 | 1 | 0.812 | 0.816 | 0.456 | 0.766 | 0.840 | FAIL |
| factorized_semantic_h1_no_int_d256 | 1 | 0.700 | 0.678 | 0.334 | 0.623 | 0.680 | FAIL |
| factorized_semantic_h1_no_int_d384 | 1 | 0.752 | 0.729 | 0.364 | 0.704 | 0.720 | FAIL |
| factorized_semantic_h1_no_int_d512 | 1 | 0.811 | 0.800 | 0.448 | 0.748 | 0.800 | FAIL |

Stage-2 result:

GATED: no h1-qualified method.

### Independent h1/h2/h4/h8/h16 causal fidelity

| method/condition | h | direction | magnitude | semantic | output | sign | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| factorized_multistep_direction_int_d512 | 1 | 0.881 | 0.873 | 0.534 | 0.835 | 0.900 | FAIL |
| oracle_factorized_pca_d512 | 1 | 0.292 | 0.383 | 0.110 | 0.263 | 0.740 | FAIL |
| oracle_joint_pca_d512 | 1 | 0.926 | 0.964 | 0.594 | 0.895 | 0.860 | FAIL |
| unified_causal_d512 | 1 | 0.929 | 0.988 | 0.630 | 0.899 | 0.820 | FAIL |
| unified_causal_d512 | 2 | 0.828 | 0.997 | 0.332 | 0.788 | 0.760 | FAIL |
| unified_causal_d512 | 4 | 0.676 | 1.005 | 0.186 | 0.607 | 0.739 | FAIL |
| unified_causal_d512 | 8 | 0.516 | 1.005 | 0.074 | 0.526 | 0.800 | FAIL |
| unified_causal_d512 | 16 | 0.541 | 1.017 | 0.102 | 0.528 | 0.653 | FAIL |

### Architecture ceiling gaps

# Architecture-resolved ceiling — V11

- Ceiling A: combined block-normalized raw dual-PCA, 599D.
- Ceiling B: separate REC + conv + KV coordinates, 1797D.
- Ceiling C: raw full persistent intervention, whose causal identity fidelity is 1 by definition.

| target | combined 599 | architecture 1797 | B − A | metric |
|---|---:|---:|---:|---|
| h1_j_effect | 0.530 | 0.642 | +0.112 | direction_cosine |
| h4_j_effect | 0.190 | 0.230 | +0.040 | direction_cosine |
| output_effect | 0.503 | 0.610 | +0.107 | correlation |

The old 599D result is therefore called a **combined-reference ceiling**, not a complete raw
persistent-state ceiling.

Machine records: `results/v11/processed/architecture_ceiling_v11.parquet` (`ef74609c1c412286a35643f199dc0c4bc03df72fdac54aaa4527f35b4cf7f41a`).


### Adjudication

- Decision-tree outcome: **C**.
- Strongest warranted conclusion: tested low-rank oracle projections do not establish writable sufficiency.
- Smallest causally validated dimension: `None`.
- Hypothesis status: **H2**.
- Autonomous-controller training authorized: **False**.
- Free continuation executed: **False**.
- Manifold record: `results/v11/processed/causal_state_manifold_v11.parquet`.
- Amplification record: `results/v11/processed/causal_error_amplification_v11.parquet`.

### Exact commands

- `bash scripts/run_causal_geometry_v11.sh freeze-base`
- `bash scripts/run_causal_geometry_v11.sh select-models`
- `bash scripts/run_causal_geometry_v11.sh prepare-development`
- `bash scripts/run_causal_geometry_v11.sh freeze-prepared`
- `bash scripts/run_causal_geometry_v11.sh channel-audit`
- `bash scripts/run_causal_geometry_v11.sh oracle-development`
- `bash scripts/run_causal_geometry_v11.sh factorized-stage1`
- `bash scripts/run_causal_geometry_v11.sh freeze-stage2`
- `bash scripts/run_causal_geometry_v11.sh factorized-stage2`
- `bash scripts/run_causal_geometry_v11.sh freeze-confirm`
- `bash scripts/run_causal_geometry_v11.sh prepare-confirmatory`
- `bash scripts/run_causal_geometry_v11.sh confirmatory`
- `bash scripts/run_causal_geometry_v11.sh analyze`
- `bash scripts/run_causal_geometry_v11.sh report`

### V11 changed/generated files

- `artifacts/causal_geometry_v11.freeze.json`
- `artifacts/causal_geometry_v11_confirmatory.freeze.json`
- `artifacts/causal_geometry_v11_manifold_amendment.freeze.json`
- `artifacts/causal_geometry_v11_prepared.freeze.json`
- `artifacts/causal_geometry_v11_stage2.freeze.json`
- `artifacts/v10_immutable.sha256.json`
- `configs/causal_geometry_v11.yaml`
- `reports/ARCH_RESOLVED_CEILING_V11.md`
- `reports/CAUSAL_ERROR_AMPLIFICATION_V11.md`
- `reports/CAUSAL_STATE_MANIFOLD_AUDIT_V11.md`
- `reports/CHANNELWISE_CAUSAL_DECODER_AUDIT_V11.md`
- `reports/FACTORIZED_CAUSAL_STATE_V11.md`
- `reports/ORACLE_LOWRANK_CAUSAL_STATE_V11.md`
- `results/v11/processed/adjudication_v11.json`
- `results/v11/processed/allocation_search_v11.parquet`
- `results/v11/processed/architecture_ceiling_v11.json`
- `results/v11/processed/architecture_ceiling_v11.parquet`
- `results/v11/processed/causal_confirmatory_v11.json`
- `results/v11/processed/causal_confirmatory_v11.parquet`
- `results/v11/processed/causal_error_amplification_v11.json`
- `results/v11/processed/causal_error_amplification_v11.parquet`
- `results/v11/processed/causal_state_manifold_v11.json`
- `results/v11/processed/causal_state_manifold_v11.parquet`
- `results/v11/processed/causal_state_manifold_v11_amendment_1.json`
- `results/v11/processed/causal_state_manifold_v11_amendment_1.parquet`
- `results/v11/processed/channel_compatibility_v11.parquet`
- `results/v11/processed/channel_interaction_nonadditivity_v11.json`
- `results/v11/processed/channel_interaction_nonadditivity_v11.parquet`
- `results/v11/processed/channelwise_causal_v11.json`
- `results/v11/processed/channelwise_causal_v11.parquet`
- `results/v11/processed/confirmatory_states_v11.json`
- `results/v11/processed/development_states_v11.json`
- `results/v11/processed/factorized_causal_stage1_v11.json`
- `results/v11/processed/factorized_causal_stage1_v11.parquet`
- `results/v11/processed/factorized_causal_stage2_v11.json`
- `results/v11/processed/factorized_causal_stage2_v11.parquet`
- `results/v11/processed/method_specs_v11.json`
- `results/v11/processed/oracle_lowrank_causal_v11.json`
- `results/v11/processed/oracle_lowrank_causal_v11.parquet`
- `results/v11/raw/analyze-v11-20260916T130456Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/analyze-v11-20260916T130911Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/causal-v11-20260916T122416Z-0f38a5ef-s20260828/channel_audit_progress.json`
- `results/v11/raw/causal-v11-20260916T122416Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/causal-v11-20260916T123047Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/causal-v11-20260916T123047Z-0f38a5ef-s20260828/oracle_lowrank_causal_projection_progress.json`
- `results/v11/raw/causal-v11-20260916T125009Z-0f38a5ef-s20260828/factorized_causal_stage1_progress.json`
- `results/v11/raw/causal-v11-20260916T125009Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/causal-v11-20260916T125529Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/causal-v11-20260916T125538Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/causal-v11-20260916T125555Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/causal-v11-20260916T125710Z-0f38a5ef-s20260828/independent_causal_confirmatory_progress.json`
- `results/v11/raw/causal-v11-20260916T125710Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/prepare-v11-20260916T121550Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/prepare-v11-20260916T121642Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/prepare-v11-20260916T121849Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/prepare-v11-20260916T121849Z-0f38a5ef-s20260828/prepare_progress.json`
- `results/v11/raw/prepare-v11-20260916T122402Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/prepare-v11-20260916T125610Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/prepare-v11-20260916T125610Z-0f38a5ef-s20260828/prepare_progress.json`
- `results/v11/raw/v11-manifold-amendment-20260916T130850Z-0f38a5ef-s20260828/manifest.json`
- `results/v11/raw/v11-manifold-amendment-20260916T130854Z-0f38a5ef-s20260828/manifest.json`
- `schemas/protocol-v11-record.schema.json`
- `scripts/run_causal_geometry_v11.sh`
- `scripts/run_causal_geometry_v11_manifold_amendment.sh`
- `src/jclosure/experiments/analyze_v11.py`
- `src/jclosure/experiments/analyze_v11_amendment.py`
- `src/jclosure/experiments/causal_v11.py`
- `src/jclosure/experiments/prepare_v11.py`
- `src/jclosure/protocol_v11.py`
- `src/jclosure/protocol_v11_manifold_amendment.py`
- `src/jclosure/reporting_v11.py`
- `src/jclosure/reporting_v11_amendment.py`
- `src/jclosure/state_models_v11.py`
- `tests/test_v11.py`
- `tests/test_v11_manifold_amendment.py`

## Standalone report integrity index

| report | SHA256 |
|---|---|
| `reports/CHANNELWISE_CAUSAL_DECODER_AUDIT_V11.md` | `68fefc323c26b516cfd7b83fe7642f5ccd8c4c3a4b739c7f44fffa94d88a9ecd` |
| `reports/ORACLE_LOWRANK_CAUSAL_STATE_V11.md` | `4758228ad7891e4cf4d799259648c62e09b789e91b3aa263acc499e040a1a42f` |
| `reports/FACTORIZED_CAUSAL_STATE_V11.md` | `97040781df67202a655d860531212436f113ae3fa2f6a6fb735fd0aece7df05e` |
| `reports/CAUSAL_STATE_MANIFOLD_AUDIT_V11.md` | `cdebd0c095aba845b19e46abd5d7485645bc548b5c6a57a6e89c57ee0902242a` |
| `reports/CAUSAL_ERROR_AMPLIFICATION_V11.md` | `d49de610fefa462e3abcacab925629c564b88cb689bbf14e0e95d47db85d22a8` |
| `reports/ARCH_RESOLVED_CEILING_V11.md` | `c0073fb60f7f1d0f89d04cda5b824163355cdec2edfb147b9a5059c772c0868f` |

## Machine-record integrity index

| record | bytes | SHA256 |
|---|---:|---|
| `results/v11/processed/adjudication_v11.json` | 642 | `8a0c0813e3548502667690ecc7cb5992a1c1125a0132c5c4eec4417049e8e631` |
| `results/v11/processed/allocation_search_v11.parquet` | 10311 | `5518373bc26b5e26d2e80f9b8f495f3dc04175ae454d7b21a76379c6dcae9a90` |
| `results/v11/processed/architecture_ceiling_v11.json` | 11108 | `4558a15d392980da06ca3f115a642fee33554abf457ba021ed97659c6df3e180` |
| `results/v11/processed/architecture_ceiling_v11.parquet` | 8174 | `ef74609c1c412286a35643f199dc0c4bc03df72fdac54aaa4527f35b4cf7f41a` |
| `results/v11/processed/causal_confirmatory_v11.json` | 269830 | `54ad0e45d37493fe1f69a904ac2c576042cf434ce9fd572c4efa1776fbe86ba1` |
| `results/v11/processed/causal_confirmatory_v11.parquet` | 55801 | `d285ab0ebc397663ba3a5882359f0b9b2370446c2ec1306061bca8042edf3065` |
| `results/v11/processed/causal_error_amplification_v11.json` | 46075 | `3c7c863bcdbc36e3b5db0c54517dd93676338fcc18a626880a59927743579f30` |
| `results/v11/processed/causal_error_amplification_v11.parquet` | 203489 | `ef9a3ccbc477ef5f4e48024a848bd9b64d4aa3c25fe621580c5e733288084887` |
| `results/v11/processed/causal_state_manifold_v11.json` | 8508 | `c8fae61c1b5a24c3dbcce49b1f39feb101836968190c97690410342a450d5935` |
| `results/v11/processed/causal_state_manifold_v11.parquet` | 43714 | `06f13b242adae1b38ac10e7e76c2401b5b20850fc13e9161a7e60efbda93465e` |
| `results/v11/processed/causal_state_manifold_v11_amendment_1.json` | 3475 | `b4dfa7e0c493465a597c39d70f3980773740018cb71191b800797f5d075f9fba` |
| `results/v11/processed/causal_state_manifold_v11_amendment_1.parquet` | 44823 | `64e60138414fe876f5140586fd6b72058d8f9c05e76b03ead499720da078017b` |
| `results/v11/processed/channel_compatibility_v11.parquet` | 8018 | `c33c0aa686277f8299d69fc08e76b9ed902ba2e455bb13e26ec9307e2e12183b` |
| `results/v11/processed/channel_interaction_nonadditivity_v11.json` | 5141 | `233bc738ac0a3f4478b581836e54f8c77d7bfec9c1643194b823a4591102412a` |
| `results/v11/processed/channel_interaction_nonadditivity_v11.parquet` | 43130 | `c8014adb7c99d2c0d04fa451d671c5828c741f81cde78c6538564a16148c3a13` |
| `results/v11/processed/channelwise_causal_v11.json` | 1059293 | `f03a2072184f5537e53fd3167367fcf43d9d84ea011e0b1baac92c6796fbadc4` |
| `results/v11/processed/channelwise_causal_v11.parquet` | 123042 | `f46ab7afc99287e6106ba6eb5ce032632d656d8e163a813b17bd583602b2063e` |
| `results/v11/processed/confirmatory_states_v11.json` | 29089 | `c2c1736c47626e7dfeb53d4357bd6e18c99e7c6ad80cdc83bd332125db789c2a` |
| `results/v11/processed/development_states_v11.json` | 435478 | `5c27e5616ba6919f9dfa75946ca4a91b88b04864a9208a2ef8e717e36f32c663` |
| `results/v11/processed/factorized_causal_stage1_v11.json` | 1621489 | `d94daf1892c7a1902faee6017a521d02481c66871d9767cdcf765b272d6e6535` |
| `results/v11/processed/factorized_causal_stage1_v11.parquet` | 172692 | `669cdecda89e4742d67a77a29228982c698d36a98854122bf9d185e1d355c0d9` |
| `results/v11/processed/factorized_causal_stage2_v11.json` | 169 | `2a24a98927de79852ff4bde6ccf3a864cefa287c5392cd7ff4f98681911e7dcc` |
| `results/v11/processed/factorized_causal_stage2_v11.parquet` | 974 | `e7172c5578d86d1624b9c689cdaf1fbaf81dbf071967187f129039465b068b2a` |
| `results/v11/processed/method_specs_v11.json` | 17195 | `8b99300538dded02d1a2eb994b0711051c29591dc4807e091303b7d00bad08e4` |
| `results/v11/processed/oracle_lowrank_causal_v11.json` | 2677891 | `6ee1ad673c9d9a6bb31edb63d6b6965eff632372134516e8e582a7a487b8e333` |
| `results/v11/processed/oracle_lowrank_causal_v11.parquet` | 291297 | `d4434d624419e318a6c8d06252f9bdd31bafbcf7aa9e236d5a6876a50e671e93` |

---

## Bundled report 1: `CHANNELWISE_CAUSAL_DECODER_AUDIT_V11.md`

# Channel-wise causal decoder audit — V11

This is a frozen development-bank diagnosis. Teacher raw channels are retained except
where the condition name says `decoded`; all trajectories are teacher-forced.

| method/condition | h | direction | magnitude | semantic | output | sign | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| decoded_all | 1 | 0.923 | 0.977 | 0.614 | 0.896 | 0.820 | FAIL |
| decoded_all | 2 | 0.823 | 0.980 | 0.352 | 0.795 | 0.740 | FAIL |
| decoded_all | 4 | 0.688 | 0.987 | 0.206 | 0.660 | 0.696 | FAIL |
| decoded_all | 8 | 0.532 | 0.998 | 0.086 | 0.528 | 0.660 | FAIL |
| decoded_conv | 1 | 0.934 | 0.984 | 0.672 | 0.904 | 0.900 | FAIL |
| decoded_conv | 2 | 0.837 | 0.998 | 0.366 | 0.806 | 0.760 | FAIL |
| decoded_conv | 4 | 0.705 | 0.989 | 0.242 | 0.676 | 0.761 | FAIL |
| decoded_conv | 8 | 0.544 | 1.010 | 0.110 | 0.512 | 0.740 | FAIL |
| decoded_conv_kv | 1 | 0.932 | 0.978 | 0.656 | 0.907 | 0.860 | FAIL |
| decoded_conv_kv | 2 | 0.840 | 0.998 | 0.384 | 0.805 | 0.740 | FAIL |
| decoded_conv_kv | 4 | 0.703 | 0.991 | 0.228 | 0.671 | 0.761 | FAIL |
| decoded_conv_kv | 8 | 0.554 | 1.011 | 0.132 | 0.544 | 0.680 | FAIL |
| decoded_kv | 1 | 0.970 | 0.995 | 0.760 | 0.945 | 0.820 | FAIL |
| decoded_kv | 2 | 0.911 | 0.997 | 0.500 | 0.874 | 0.800 | FAIL |
| decoded_kv | 4 | 0.861 | 0.989 | 0.406 | 0.807 | 0.783 | FAIL |
| decoded_kv | 8 | 0.770 | 1.010 | 0.266 | 0.660 | 0.740 | FAIL |
| decoded_rec | 1 | 0.942 | 0.997 | 0.674 | 0.913 | 0.940 | FAIL |
| decoded_rec | 2 | 0.833 | 0.995 | 0.382 | 0.818 | 0.740 | FAIL |
| decoded_rec | 4 | 0.697 | 0.993 | 0.222 | 0.628 | 0.804 | FAIL |
| decoded_rec | 8 | 0.530 | 1.003 | 0.096 | 0.527 | 0.700 | FAIL |
| decoded_rec_conv | 1 | 0.923 | 0.979 | 0.624 | 0.900 | 0.860 | FAIL |
| decoded_rec_conv | 2 | 0.826 | 0.991 | 0.352 | 0.802 | 0.760 | FAIL |
| decoded_rec_conv | 4 | 0.691 | 0.987 | 0.212 | 0.652 | 0.565 | FAIL |
| decoded_rec_conv | 8 | 0.529 | 1.004 | 0.072 | 0.519 | 0.620 | FAIL |
| decoded_rec_kv | 1 | 0.941 | 1.000 | 0.682 | 0.914 | 0.940 | FAIL |
| decoded_rec_kv | 2 | 0.832 | 0.998 | 0.404 | 0.814 | 0.800 | FAIL |
| decoded_rec_kv | 4 | 0.697 | 0.995 | 0.212 | 0.655 | 0.696 | FAIL |
| decoded_rec_kv | 8 | 0.527 | 1.006 | 0.080 | 0.517 | 0.720 | FAIL |
| teacher_reference | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| teacher_reference | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| teacher_reference | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |
| teacher_reference | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | PASS |

The weakest single decoded channel at h1 is **decoded_conv**. The weakest overall decoded
combination at h1 is **decoded_all**. Non-additivity is assessed in the amplification machine
record rather than inferred from marginal reconstruction error.

Machine records: `results/v11/processed/channelwise_causal_v11.parquet` (`f46ab7afc99287e6106ba6eb5ce032632d656d8e163a813b17bd583602b2063e`).


<!-- V11_MANIFOLD_AMENDMENT_1 -->
## Channel nonadditivity amendment 1

The retained records do not contain full error vectors, so exact cross terms cannot be
recovered. The frozen descriptive proxy compares observed joint error norm with the
root-sum-square of single-channel error norms. Ratio >1 is compatible with constructive
interaction; ratio <1 with cancellation.

| decoded channels | h | joint error | single RSS | joint/RSS |
|---|---:|---:|---:|---:|
| all | 1 | 0.0089 | 0.0126 | 0.706 |
| all | 2 | 0.0065 | 0.0102 | 0.643 |
| all | 4 | 0.0120 | 0.0185 | 0.649 |
| all | 8 | 0.0072 | 0.0114 | 0.633 |
| conv_kv | 1 | 0.0084 | 0.0100 | 0.843 |
| conv_kv | 2 | 0.0063 | 0.0079 | 0.797 |
| conv_kv | 4 | 0.0117 | 0.0142 | 0.822 |
| conv_kv | 8 | 0.0071 | 0.0088 | 0.807 |
| rec_conv | 1 | 0.0089 | 0.0113 | 0.785 |
| rec_conv | 2 | 0.0065 | 0.0090 | 0.721 |
| rec_conv | 4 | 0.0120 | 0.0167 | 0.721 |
| rec_conv | 8 | 0.0072 | 0.0102 | 0.713 |
| rec_kv | 1 | 0.0076 | 0.0094 | 0.816 |
| rec_kv | 2 | 0.0064 | 0.0079 | 0.813 |
| rec_kv | 4 | 0.0119 | 0.0144 | 0.830 |
| rec_kv | 8 | 0.0073 | 0.0089 | 0.821 |

Amendment freeze: `f17f62532f1136c9cc73109ecf55e2f6689b55638313979eb9c1d5068ac5c14f`. Machine records: `results/v11/processed/channel_interaction_nonadditivity_v11.parquet` (`c8014adb7c99d2c0d04fa451d671c5828c741f81cde78c6538564a16148c3a13`).

---

## Bundled report 2: `ORACLE_LOWRANK_CAUSAL_STATE_V11.md`

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

---

## Bundled report 3: `FACTORIZED_CAUSAL_STATE_V11.md`

# Factorized causal state — V11

Allocation and objective selection used validation/development data only. The tested
objectives cover cache PCA, h1 direction, semantic effect, output effect, multistep effect,
effect weighting, a composite causal objective, and variance-based manifold regularization.

## Stage 1: h1

| method/condition | h | direction | magnitude | semantic | output | sign | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| factorized_cache_pca_int_d256 | 1 | 0.179 | 0.331 | 0.074 | 0.236 | 0.620 | FAIL |
| factorized_cache_pca_int_d384 | 1 | 0.207 | 0.344 | 0.072 | 0.210 | 0.440 | FAIL |
| factorized_cache_pca_int_d512 | 1 | 0.289 | 0.384 | 0.084 | 0.314 | 0.600 | FAIL |
| factorized_cache_pca_no_int_d256 | 1 | 0.173 | 0.342 | 0.064 | 0.237 | 0.560 | FAIL |
| factorized_cache_pca_no_int_d384 | 1 | 0.227 | 0.345 | 0.082 | 0.259 | 0.580 | FAIL |
| factorized_cache_pca_no_int_d512 | 1 | 0.346 | 0.409 | 0.102 | 0.293 | 0.580 | FAIL |
| factorized_causal_composite_int_d256 | 1 | 0.745 | 0.744 | 0.386 | 0.679 | 0.680 | FAIL |
| factorized_causal_composite_int_d384 | 1 | 0.762 | 0.752 | 0.396 | 0.692 | 0.680 | FAIL |
| factorized_causal_composite_int_d512 | 1 | 0.816 | 0.818 | 0.432 | 0.770 | 0.740 | FAIL |
| factorized_causal_composite_no_int_d256 | 1 | 0.699 | 0.676 | 0.348 | 0.611 | 0.680 | FAIL |
| factorized_causal_composite_no_int_d384 | 1 | 0.743 | 0.730 | 0.376 | 0.709 | 0.640 | FAIL |
| factorized_causal_composite_no_int_d512 | 1 | 0.808 | 0.807 | 0.464 | 0.749 | 0.760 | FAIL |
| factorized_effect_weighted_multistep_int_d256 | 1 | 0.792 | 0.809 | 0.414 | 0.770 | 0.660 | FAIL |
| factorized_effect_weighted_multistep_int_d384 | 1 | 0.833 | 0.825 | 0.494 | 0.789 | 0.720 | FAIL |
| factorized_effect_weighted_multistep_int_d512 | 1 | 0.849 | 0.855 | 0.492 | 0.821 | 0.740 | FAIL |
| factorized_effect_weighted_multistep_no_int_d256 | 1 | 0.774 | 0.749 | 0.416 | 0.721 | 0.700 | FAIL |
| factorized_effect_weighted_multistep_no_int_d384 | 1 | 0.815 | 0.784 | 0.446 | 0.771 | 0.680 | FAIL |
| factorized_effect_weighted_multistep_no_int_d512 | 1 | 0.840 | 0.832 | 0.508 | 0.805 | 0.720 | FAIL |
| factorized_h1_direction_int_d256 | 1 | 0.803 | 0.815 | 0.430 | 0.772 | 0.800 | FAIL |
| factorized_h1_direction_int_d384 | 1 | 0.835 | 0.857 | 0.454 | 0.804 | 0.700 | FAIL |
| factorized_h1_direction_int_d512 | 1 | 0.852 | 0.875 | 0.522 | 0.821 | 0.740 | FAIL |
| factorized_h1_direction_no_int_d256 | 1 | 0.760 | 0.730 | 0.376 | 0.713 | 0.740 | FAIL |
| factorized_h1_direction_no_int_d384 | 1 | 0.817 | 0.797 | 0.470 | 0.776 | 0.760 | FAIL |
| factorized_h1_direction_no_int_d512 | 1 | 0.845 | 0.853 | 0.518 | 0.802 | 0.800 | FAIL |
| factorized_manifold_regularized_int_d256 | 1 | 0.761 | 0.750 | 0.374 | 0.707 | 0.740 | FAIL |
| factorized_manifold_regularized_int_d384 | 1 | 0.762 | 0.752 | 0.396 | 0.692 | 0.680 | FAIL |
| factorized_manifold_regularized_int_d512 | 1 | 0.816 | 0.818 | 0.432 | 0.770 | 0.740 | FAIL |
| factorized_manifold_regularized_no_int_d256 | 1 | 0.699 | 0.676 | 0.348 | 0.611 | 0.680 | FAIL |
| factorized_manifold_regularized_no_int_d384 | 1 | 0.751 | 0.725 | 0.366 | 0.664 | 0.700 | FAIL |
| factorized_manifold_regularized_no_int_d512 | 1 | 0.808 | 0.807 | 0.464 | 0.749 | 0.760 | FAIL |
| factorized_multistep_direction_int_d256 | 1 | 0.790 | 0.784 | 0.432 | 0.749 | 0.720 | FAIL |
| factorized_multistep_direction_int_d384 | 1 | 0.836 | 0.841 | 0.472 | 0.790 | 0.720 | FAIL |
| factorized_multistep_direction_int_d512 | 1 | 0.859 | 0.880 | 0.516 | 0.827 | 0.760 | FAIL |
| factorized_multistep_direction_no_int_d256 | 1 | 0.760 | 0.732 | 0.418 | 0.729 | 0.720 | FAIL |
| factorized_multistep_direction_no_int_d384 | 1 | 0.808 | 0.798 | 0.478 | 0.767 | 0.800 | FAIL |
| factorized_multistep_direction_no_int_d512 | 1 | 0.849 | 0.847 | 0.508 | 0.810 | 0.780 | FAIL |
| factorized_output_h1_int_d256 | 1 | 0.657 | 0.651 | 0.318 | 0.601 | 0.660 | FAIL |
| factorized_output_h1_int_d384 | 1 | 0.705 | 0.712 | 0.350 | 0.655 | 0.760 | FAIL |
| factorized_output_h1_int_d512 | 1 | 0.727 | 0.705 | 0.364 | 0.665 | 0.740 | FAIL |
| factorized_output_h1_no_int_d256 | 1 | 0.650 | 0.629 | 0.278 | 0.589 | 0.700 | FAIL |
| factorized_output_h1_no_int_d384 | 1 | 0.710 | 0.685 | 0.356 | 0.650 | 0.700 | FAIL |
| factorized_output_h1_no_int_d512 | 1 | 0.718 | 0.691 | 0.376 | 0.651 | 0.720 | FAIL |
| factorized_semantic_h1_int_d256 | 1 | 0.766 | 0.758 | 0.394 | 0.716 | 0.640 | FAIL |
| factorized_semantic_h1_int_d384 | 1 | 0.756 | 0.757 | 0.392 | 0.688 | 0.640 | FAIL |
| factorized_semantic_h1_int_d512 | 1 | 0.812 | 0.816 | 0.456 | 0.766 | 0.840 | FAIL |
| factorized_semantic_h1_no_int_d256 | 1 | 0.700 | 0.678 | 0.334 | 0.623 | 0.680 | FAIL |
| factorized_semantic_h1_no_int_d384 | 1 | 0.752 | 0.729 | 0.364 | 0.704 | 0.720 | FAIL |
| factorized_semantic_h1_no_int_d512 | 1 | 0.811 | 0.800 | 0.448 | 0.748 | 0.800 | FAIL |

## Stage 2: h1/h2/h4

No factorized method passed the frozen all-family h1 gate; h2/h4 was gated.

Frozen confirmatory methods: `['unified_causal_d512', 'oracle_joint_pca_d512', 'oracle_factorized_pca_d512', 'factorized_multistep_direction_int_d512']`.

## Independent confirmation

| method/condition | h | direction | magnitude | semantic | output | sign | gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| factorized_multistep_direction_int_d512 | 1 | 0.881 | 0.873 | 0.534 | 0.835 | 0.900 | FAIL |
| oracle_factorized_pca_d512 | 1 | 0.292 | 0.383 | 0.110 | 0.263 | 0.740 | FAIL |
| oracle_joint_pca_d512 | 1 | 0.926 | 0.964 | 0.594 | 0.895 | 0.860 | FAIL |
| unified_causal_d512 | 1 | 0.929 | 0.988 | 0.630 | 0.899 | 0.820 | FAIL |
| unified_causal_d512 | 2 | 0.828 | 0.997 | 0.332 | 0.788 | 0.760 | FAIL |
| unified_causal_d512 | 4 | 0.676 | 1.005 | 0.186 | 0.607 | 0.739 | FAIL |
| unified_causal_d512 | 8 | 0.516 | 1.005 | 0.074 | 0.526 | 0.800 | FAIL |
| unified_causal_d512 | 16 | 0.541 | 1.017 | 0.102 | 0.528 | 0.653 | FAIL |

---

## Bundled report 4: `CAUSAL_STATE_MANIFOLD_AUDIT_V11.md`

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

---

## Bundled report 5: `CAUSAL_ERROR_AMPLIFICATION_V11.md`

# Causal error amplification — V11

`e_h` is recovered exactly from teacher/decoded effect norms and their cosine. Ratios use
the next measured horizon (1→2→4→8→16), so they are finite-horizon amplification proxies,
not one-step Jacobian eigenvalues. Explicit JVP/Jacobian estimation was not run.

| source:method | h | error norm | direction error | semantic error | ratio from prior h |
|---|---:|---:|---:|---:|---:|
| channel:decoded_all | 1 | 0.009 | 0.077 | 0.386 | NA |
| channel:decoded_all | 2 | 0.007 | 0.177 | 0.648 | 0.743 |
| channel:decoded_all | 4 | 0.012 | 0.312 | 0.794 | 1.866 |
| channel:decoded_all | 8 | 0.007 | 0.468 | 0.914 | 0.716 |
| channel:decoded_conv | 1 | 0.008 | 0.066 | 0.328 | NA |
| channel:decoded_conv | 2 | 0.006 | 0.163 | 0.634 | 0.771 |
| channel:decoded_conv | 4 | 0.012 | 0.295 | 0.758 | 1.866 |
| channel:decoded_conv | 8 | 0.007 | 0.456 | 0.890 | 0.740 |
| channel:decoded_conv_kv | 1 | 0.008 | 0.068 | 0.344 | NA |
| channel:decoded_conv_kv | 2 | 0.006 | 0.160 | 0.616 | 0.757 |
| channel:decoded_conv_kv | 4 | 0.012 | 0.297 | 0.772 | 1.880 |
| channel:decoded_conv_kv | 8 | 0.007 | 0.446 | 0.868 | 0.731 |
| channel:decoded_kv | 1 | 0.005 | 0.030 | 0.240 | NA |
| channel:decoded_kv | 2 | 0.005 | 0.089 | 0.500 | 0.863 |
| channel:decoded_kv | 4 | 0.008 | 0.139 | 0.594 | 1.728 |
| channel:decoded_kv | 8 | 0.005 | 0.230 | 0.734 | 0.717 |
| channel:decoded_rec | 1 | 0.008 | 0.058 | 0.326 | NA |
| channel:decoded_rec | 2 | 0.006 | 0.167 | 0.618 | 0.850 |
| channel:decoded_rec | 4 | 0.012 | 0.303 | 0.778 | 1.884 |
| channel:decoded_rec | 8 | 0.007 | 0.470 | 0.904 | 0.736 |
| channel:decoded_rec_conv | 1 | 0.009 | 0.077 | 0.376 | NA |
| channel:decoded_rec_conv | 2 | 0.006 | 0.174 | 0.648 | 0.741 |
| channel:decoded_rec_conv | 4 | 0.012 | 0.309 | 0.788 | 1.863 |
| channel:decoded_rec_conv | 8 | 0.007 | 0.471 | 0.928 | 0.722 |
| channel:decoded_rec_kv | 1 | 0.008 | 0.059 | 0.318 | NA |
| channel:decoded_rec_kv | 2 | 0.006 | 0.168 | 0.596 | 0.852 |
| channel:decoded_rec_kv | 4 | 0.012 | 0.303 | 0.788 | 1.867 |
| channel:decoded_rec_kv | 8 | 0.007 | 0.473 | 0.920 | 0.723 |
| channel:teacher_reference | 1 | 0.000 | -0.000 | 0.000 | NA |
| channel:teacher_reference | 2 | 0.000 | -0.000 | 0.000 | 56565960027176.398 |
| channel:teacher_reference | 4 | 0.000 | -0.000 | 0.000 | 142340500221486.375 |
| channel:teacher_reference | 8 | 0.000 | 0.000 | 0.000 | 70515751372593.734 |
| confirmatory:factorized_multistep_direction_int_d512 | 1 | 0.012 | 0.119 | 0.466 | NA |
| confirmatory:oracle_factorized_pca_d512 | 1 | 0.024 | 0.708 | 0.890 | NA |
| confirmatory:oracle_joint_pca_d512 | 1 | 0.009 | 0.074 | 0.406 | NA |
| confirmatory:unified_causal_d512 | 1 | 0.009 | 0.071 | 0.370 | NA |
| confirmatory:unified_causal_d512 | 2 | 0.007 | 0.172 | 0.668 | 0.735 |
| confirmatory:unified_causal_d512 | 4 | 0.012 | 0.324 | 0.814 | 1.826 |
| confirmatory:unified_causal_d512 | 8 | 0.008 | 0.484 | 0.926 | 0.785 |
| confirmatory:unified_causal_d512 | 16 | 0.006 | 0.459 | 0.898 | 0.762 |
| factorized_stage1:factorized_cache_pca_int_d256 | 1 | 0.025 | 0.821 | 0.926 | NA |
| factorized_stage1:factorized_cache_pca_int_d384 | 1 | 0.024 | 0.793 | 0.928 | NA |
| factorized_stage1:factorized_cache_pca_int_d512 | 1 | 0.024 | 0.711 | 0.916 | NA |
| factorized_stage1:factorized_cache_pca_no_int_d256 | 1 | 0.025 | 0.827 | 0.936 | NA |
| factorized_stage1:factorized_cache_pca_no_int_d384 | 1 | 0.024 | 0.773 | 0.918 | NA |
| factorized_stage1:factorized_cache_pca_no_int_d512 | 1 | 0.023 | 0.654 | 0.898 | NA |
| factorized_stage1:factorized_causal_composite_int_d256 | 1 | 0.016 | 0.255 | 0.614 | NA |
| factorized_stage1:factorized_causal_composite_int_d384 | 1 | 0.016 | 0.238 | 0.604 | NA |
| factorized_stage1:factorized_causal_composite_int_d512 | 1 | 0.014 | 0.184 | 0.568 | NA |
| factorized_stage1:factorized_causal_composite_no_int_d256 | 1 | 0.017 | 0.301 | 0.652 | NA |
| factorized_stage1:factorized_causal_composite_no_int_d384 | 1 | 0.016 | 0.257 | 0.624 | NA |
| factorized_stage1:factorized_causal_composite_no_int_d512 | 1 | 0.014 | 0.192 | 0.536 | NA |
| factorized_stage1:factorized_effect_weighted_multistep_int_d256 | 1 | 0.014 | 0.208 | 0.586 | NA |
| factorized_stage1:factorized_effect_weighted_multistep_int_d384 | 1 | 0.013 | 0.167 | 0.506 | NA |
| factorized_stage1:factorized_effect_weighted_multistep_int_d512 | 1 | 0.013 | 0.151 | 0.508 | NA |
| factorized_stage1:factorized_effect_weighted_multistep_no_int_d256 | 1 | 0.015 | 0.226 | 0.584 | NA |
| factorized_stage1:factorized_effect_weighted_multistep_no_int_d384 | 1 | 0.014 | 0.185 | 0.554 | NA |
| factorized_stage1:factorized_effect_weighted_multistep_no_int_d512 | 1 | 0.013 | 0.160 | 0.492 | NA |
| factorized_stage1:factorized_h1_direction_int_d256 | 1 | 0.014 | 0.197 | 0.570 | NA |
| factorized_stage1:factorized_h1_direction_int_d384 | 1 | 0.013 | 0.165 | 0.546 | NA |
| factorized_stage1:factorized_h1_direction_int_d512 | 1 | 0.012 | 0.148 | 0.478 | NA |
| factorized_stage1:factorized_h1_direction_no_int_d256 | 1 | 0.015 | 0.240 | 0.624 | NA |
| factorized_stage1:factorized_h1_direction_no_int_d384 | 1 | 0.014 | 0.183 | 0.530 | NA |
| factorized_stage1:factorized_h1_direction_no_int_d512 | 1 | 0.013 | 0.155 | 0.482 | NA |
| factorized_stage1:factorized_manifold_regularized_int_d256 | 1 | 0.015 | 0.239 | 0.626 | NA |
| factorized_stage1:factorized_manifold_regularized_int_d384 | 1 | 0.016 | 0.238 | 0.604 | NA |
| factorized_stage1:factorized_manifold_regularized_int_d512 | 1 | 0.014 | 0.184 | 0.568 | NA |
| factorized_stage1:factorized_manifold_regularized_no_int_d256 | 1 | 0.017 | 0.301 | 0.652 | NA |
| factorized_stage1:factorized_manifold_regularized_no_int_d384 | 1 | 0.016 | 0.249 | 0.634 | NA |
| factorized_stage1:factorized_manifold_regularized_no_int_d512 | 1 | 0.014 | 0.192 | 0.536 | NA |
| factorized_stage1:factorized_multistep_direction_int_d256 | 1 | 0.014 | 0.210 | 0.568 | NA |
| factorized_stage1:factorized_multistep_direction_int_d384 | 1 | 0.013 | 0.164 | 0.528 | NA |
| factorized_stage1:factorized_multistep_direction_int_d512 | 1 | 0.012 | 0.141 | 0.484 | NA |
| factorized_stage1:factorized_multistep_direction_no_int_d256 | 1 | 0.016 | 0.240 | 0.582 | NA |
| factorized_stage1:factorized_multistep_direction_no_int_d384 | 1 | 0.014 | 0.192 | 0.522 | NA |
| factorized_stage1:factorized_multistep_direction_no_int_d512 | 1 | 0.012 | 0.151 | 0.492 | NA |
| factorized_stage1:factorized_output_h1_int_d256 | 1 | 0.018 | 0.343 | 0.682 | NA |
| factorized_stage1:factorized_output_h1_int_d384 | 1 | 0.017 | 0.295 | 0.650 | NA |
| factorized_stage1:factorized_output_h1_int_d512 | 1 | 0.016 | 0.273 | 0.636 | NA |
| factorized_stage1:factorized_output_h1_no_int_d256 | 1 | 0.018 | 0.350 | 0.722 | NA |
| factorized_stage1:factorized_output_h1_no_int_d384 | 1 | 0.017 | 0.290 | 0.644 | NA |
| factorized_stage1:factorized_output_h1_no_int_d512 | 1 | 0.017 | 0.282 | 0.624 | NA |
| factorized_stage1:factorized_semantic_h1_int_d256 | 1 | 0.015 | 0.234 | 0.606 | NA |
| factorized_stage1:factorized_semantic_h1_int_d384 | 1 | 0.016 | 0.244 | 0.608 | NA |
| factorized_stage1:factorized_semantic_h1_int_d512 | 1 | 0.014 | 0.188 | 0.544 | NA |
| factorized_stage1:factorized_semantic_h1_no_int_d256 | 1 | 0.017 | 0.300 | 0.666 | NA |
| factorized_stage1:factorized_semantic_h1_no_int_d384 | 1 | 0.016 | 0.248 | 0.636 | NA |
| factorized_stage1:factorized_semantic_h1_no_int_d512 | 1 | 0.014 | 0.189 | 0.552 | NA |
| oracle:oracle_factorized_pca_d128 | 1 | 0.025 | 0.818 | 0.952 | NA |
| oracle:oracle_factorized_pca_d128 | 2 | 0.012 | 0.700 | 0.956 | 0.486 |
| oracle:oracle_factorized_pca_d128 | 4 | 0.016 | 0.611 | 0.938 | 1.363 |
| oracle:oracle_factorized_pca_d128 | 8 | 0.007 | 0.517 | 0.914 | 0.528 |
| oracle:oracle_factorized_pca_d128 | 16 | 0.006 | 0.535 | 0.934 | 0.839 |
| oracle:oracle_factorized_pca_d192 | 1 | 0.025 | 0.816 | 0.960 | NA |
| oracle:oracle_factorized_pca_d192 | 2 | 0.011 | 0.688 | 0.960 | 0.485 |
| oracle:oracle_factorized_pca_d192 | 4 | 0.015 | 0.600 | 0.938 | 1.348 |
| oracle:oracle_factorized_pca_d192 | 8 | 0.008 | 0.520 | 0.926 | 0.530 |
| oracle:oracle_factorized_pca_d192 | 16 | 0.006 | 0.532 | 0.914 | 0.836 |
| oracle:oracle_factorized_pca_d256 | 1 | 0.025 | 0.827 | 0.936 | NA |
| oracle:oracle_factorized_pca_d256 | 2 | 0.011 | 0.692 | 0.922 | 0.485 |
| oracle:oracle_factorized_pca_d256 | 4 | 0.015 | 0.593 | 0.928 | 1.341 |
| oracle:oracle_factorized_pca_d256 | 8 | 0.007 | 0.513 | 0.910 | 0.530 |
| oracle:oracle_factorized_pca_d256 | 16 | 0.006 | 0.539 | 0.930 | 0.843 |
| oracle:oracle_factorized_pca_d320 | 1 | 0.024 | 0.789 | 0.946 | NA |
| oracle:oracle_factorized_pca_d320 | 2 | 0.011 | 0.685 | 0.968 | 0.488 |
| oracle:oracle_factorized_pca_d320 | 4 | 0.015 | 0.599 | 0.942 | 1.355 |
| oracle:oracle_factorized_pca_d320 | 8 | 0.008 | 0.531 | 0.938 | 0.538 |
| oracle:oracle_factorized_pca_d320 | 16 | 0.006 | 0.535 | 0.910 | 0.827 |
| oracle:oracle_factorized_pca_d384 | 1 | 0.024 | 0.773 | 0.918 | NA |
| oracle:oracle_factorized_pca_d384 | 2 | 0.011 | 0.669 | 0.942 | 0.490 |
| oracle:oracle_factorized_pca_d384 | 4 | 0.015 | 0.586 | 0.946 | 1.349 |
| oracle:oracle_factorized_pca_d384 | 8 | 0.008 | 0.519 | 0.926 | 0.537 |
| oracle:oracle_factorized_pca_d384 | 16 | 0.006 | 0.536 | 0.888 | 0.840 |
| oracle:oracle_factorized_pca_d448 | 1 | 0.024 | 0.754 | 0.926 | NA |
| oracle:oracle_factorized_pca_d448 | 2 | 0.011 | 0.640 | 0.932 | 0.487 |
| oracle:oracle_factorized_pca_d448 | 4 | 0.015 | 0.562 | 0.922 | 1.355 |
| oracle:oracle_factorized_pca_d448 | 8 | 0.007 | 0.517 | 0.926 | 0.545 |
| oracle:oracle_factorized_pca_d448 | 16 | 0.006 | 0.518 | 0.904 | 0.827 |
| oracle:oracle_factorized_pca_d512 | 1 | 0.023 | 0.654 | 0.898 | NA |
| oracle:oracle_factorized_pca_d512 | 2 | 0.011 | 0.580 | 0.918 | 0.487 |
| oracle:oracle_factorized_pca_d512 | 4 | 0.015 | 0.534 | 0.932 | 1.374 |
| oracle:oracle_factorized_pca_d512 | 8 | 0.007 | 0.514 | 0.906 | 0.558 |
| oracle:oracle_factorized_pca_d512 | 16 | 0.006 | 0.515 | 0.922 | 0.829 |
| oracle:oracle_factorized_pca_d64 | 1 | 0.025 | 0.828 | 0.930 | NA |
| oracle:oracle_factorized_pca_d64 | 2 | 0.011 | 0.693 | 0.954 | 0.486 |
| oracle:oracle_factorized_pca_d64 | 4 | 0.015 | 0.608 | 0.944 | 1.366 |
| oracle:oracle_factorized_pca_d64 | 8 | 0.007 | 0.512 | 0.900 | 0.525 |
| oracle:oracle_factorized_pca_d64 | 16 | 0.006 | 0.534 | 0.918 | 0.840 |
| oracle:oracle_joint_pca_d128 | 1 | 0.024 | 0.785 | 0.924 | NA |
| oracle:oracle_joint_pca_d128 | 2 | 0.011 | 0.679 | 0.970 | 0.486 |
| oracle:oracle_joint_pca_d128 | 4 | 0.015 | 0.593 | 0.940 | 1.356 |
| oracle:oracle_joint_pca_d128 | 8 | 0.008 | 0.525 | 0.908 | 0.541 |
| oracle:oracle_joint_pca_d128 | 16 | 0.006 | 0.525 | 0.932 | 0.822 |
| oracle:oracle_joint_pca_d192 | 1 | 0.024 | 0.697 | 0.898 | NA |
| oracle:oracle_joint_pca_d192 | 2 | 0.011 | 0.632 | 0.960 | 0.488 |
| oracle:oracle_joint_pca_d192 | 4 | 0.015 | 0.576 | 0.942 | 1.379 |
| oracle:oracle_joint_pca_d192 | 8 | 0.008 | 0.526 | 0.908 | 0.543 |
| oracle:oracle_joint_pca_d192 | 16 | 0.006 | 0.524 | 0.916 | 0.827 |
| oracle:oracle_joint_pca_d256 | 1 | 0.023 | 0.590 | 0.830 | NA |
| oracle:oracle_joint_pca_d256 | 2 | 0.011 | 0.578 | 0.900 | 0.496 |
| oracle:oracle_joint_pca_d256 | 4 | 0.015 | 0.552 | 0.924 | 1.395 |
| oracle:oracle_joint_pca_d256 | 8 | 0.007 | 0.515 | 0.916 | 0.548 |
| oracle:oracle_joint_pca_d256 | 16 | 0.006 | 0.519 | 0.914 | 0.830 |
| oracle:oracle_joint_pca_d320 | 1 | 0.017 | 0.283 | 0.638 | NA |
| oracle:oracle_joint_pca_d320 | 2 | 0.009 | 0.383 | 0.822 | 0.557 |
| oracle:oracle_joint_pca_d320 | 4 | 0.014 | 0.455 | 0.884 | 1.540 |
| oracle:oracle_joint_pca_d320 | 8 | 0.007 | 0.503 | 0.898 | 0.588 |
| oracle:oracle_joint_pca_d320 | 16 | 0.006 | 0.498 | 0.908 | 0.826 |
| oracle:oracle_joint_pca_d384 | 1 | 0.012 | 0.128 | 0.464 | NA |
| oracle:oracle_joint_pca_d384 | 2 | 0.007 | 0.232 | 0.700 | 0.656 |
| oracle:oracle_joint_pca_d384 | 4 | 0.013 | 0.355 | 0.814 | 1.751 |
| oracle:oracle_joint_pca_d384 | 8 | 0.007 | 0.487 | 0.882 | 0.667 |
| oracle:oracle_joint_pca_d384 | 16 | 0.006 | 0.487 | 0.908 | 0.832 |
| oracle:oracle_joint_pca_d448 | 1 | 0.009 | 0.085 | 0.364 | NA |
| oracle:oracle_joint_pca_d448 | 2 | 0.007 | 0.189 | 0.668 | 0.729 |
| oracle:oracle_joint_pca_d448 | 4 | 0.012 | 0.326 | 0.816 | 1.834 |
| oracle:oracle_joint_pca_d448 | 8 | 0.007 | 0.477 | 0.902 | 0.706 |
| oracle:oracle_joint_pca_d448 | 16 | 0.006 | 0.468 | 0.886 | 0.818 |
| oracle:oracle_joint_pca_d512 | 1 | 0.009 | 0.079 | 0.382 | NA |
| oracle:oracle_joint_pca_d512 | 2 | 0.007 | 0.184 | 0.674 | 0.751 |
| oracle:oracle_joint_pca_d512 | 4 | 0.012 | 0.316 | 0.760 | 1.846 |
| oracle:oracle_joint_pca_d512 | 8 | 0.007 | 0.477 | 0.882 | 0.704 |
| oracle:oracle_joint_pca_d512 | 16 | 0.006 | 0.466 | 0.866 | 0.834 |
| oracle:oracle_joint_pca_d64 | 1 | 0.025 | 0.823 | 0.912 | NA |
| oracle:oracle_joint_pca_d64 | 2 | 0.011 | 0.691 | 0.944 | 0.487 |
| oracle:oracle_joint_pca_d64 | 4 | 0.015 | 0.604 | 0.938 | 1.347 |
| oracle:oracle_joint_pca_d64 | 8 | 0.008 | 0.521 | 0.908 | 0.531 |
| oracle:oracle_joint_pca_d64 | 16 | 0.006 | 0.516 | 0.934 | 0.824 |

Machine records: `results/v11/processed/causal_error_amplification_v11.parquet` (`ef9a3ccbc477ef5f4e48024a848bd9b64d4aa3c25fe621580c5e733288084887`).

---

## Bundled report 6: `ARCH_RESOLVED_CEILING_V11.md`

# Architecture-resolved ceiling — V11

- Ceiling A: combined block-normalized raw dual-PCA, 599D.
- Ceiling B: separate REC + conv + KV coordinates, 1797D.
- Ceiling C: raw full persistent intervention, whose causal identity fidelity is 1 by definition.

| target | combined 599 | architecture 1797 | B − A | metric |
|---|---:|---:|---:|---|
| h1_j_effect | 0.530 | 0.642 | +0.112 | direction_cosine |
| h4_j_effect | 0.190 | 0.230 | +0.040 | direction_cosine |
| output_effect | 0.503 | 0.610 | +0.107 | correlation |

The old 599D result is therefore called a **combined-reference ceiling**, not a complete raw
persistent-state ceiling.

Machine records: `results/v11/processed/architecture_ceiling_v11.parquet` (`ef74609c1c412286a35643f199dc0c4bc03df72fdac54aaa4527f35b4cf7f41a`).
