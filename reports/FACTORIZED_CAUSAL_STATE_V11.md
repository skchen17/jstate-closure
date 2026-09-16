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
