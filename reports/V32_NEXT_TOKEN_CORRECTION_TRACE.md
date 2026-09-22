# Next-Token Correction Trace — V32

On the frozen 10+10 tracing states, one of six prewrite probes was recorded across all 32 post-block residual outputs and 24 recurrent layers' qkv projection, raw gate-a/b projection, normalized recurrent read and recurrent output. The primary causal endpoints remain six-probe signatures.

| role | architecture boundary | rows | median conditional norm | median boundary residual alignment |
|---|---|---|---|---|
| development | gate_a | 240 | 1.084 | 0.948 |
| development | gate_b | 240 | 0.707 | 0.955 |
| development | normalized_recurrent_read | 240 | 1.470 | 0.968 |
| development | post_block_residual | 320 | 1.852 | 0.927 |
| development | qkv_projection | 240 | 12.041 | 0.921 |
| development | recurrent_output | 240 | 0.802 | 0.968 |
| validation | gate_a | 240 | 1.080 | 0.969 |
| validation | gate_b | 240 | 0.678 | 0.960 |
| validation | normalized_recurrent_read | 240 | 1.353 | 0.974 |
| validation | post_block_residual | 320 | 1.682 | 0.949 |
| validation | qkv_projection | 240 | 12.012 | 0.949 |
| validation | recurrent_output | 240 | 0.758 | 0.975 |

Layerwise post-block growth is reported below; norms and cosines are descriptive and are not donor-fidelity interventions.

| role | layer | median conditional norm | boundary target cosine | final residual target cosine |
|---|---|---|---|---|
| development | 0 | 0.110 | 1.000 | -0.013 |
| development | 1 | 0.167 | 1.000 | -0.000 |
| development | 2 | 0.406 | 1.000 | 0.006 |
| development | 3 | 0.516 | 0.907 | -0.002 |
| development | 4 | 0.660 | 0.910 | 0.004 |
| development | 5 | 0.736 | 0.916 | 0.000 |
| development | 6 | 0.907 | 0.952 | -0.007 |
| development | 7 | 0.949 | 0.920 | 0.008 |
| development | 8 | 0.948 | 0.934 | -0.005 |
| development | 9 | 1.056 | 0.954 | -0.002 |
| development | 10 | 1.126 | 0.957 | -0.010 |
| development | 11 | 1.247 | 0.946 | -0.001 |
| development | 12 | 1.274 | 0.944 | 0.002 |
| development | 13 | 1.502 | 0.956 | -0.006 |
| development | 14 | 1.533 | 0.961 | 0.021 |
| development | 15 | 1.679 | 0.941 | 0.029 |
| development | 16 | 1.743 | 0.946 | 0.024 |
| development | 17 | 1.936 | 0.960 | 0.052 |
| development | 18 | 2.211 | 0.965 | 0.073 |
| development | 19 | 2.618 | 0.887 | 0.097 |
| development | 20 | 2.929 | 0.908 | 0.123 |
| development | 21 | 3.221 | 0.909 | 0.144 |
| development | 22 | 3.434 | 0.908 | 0.175 |
| development | 23 | 3.692 | 0.833 | 0.221 |
| development | 24 | 3.780 | 0.848 | 0.238 |
| development | 25 | 3.952 | 0.858 | 0.268 |
| development | 26 | 4.202 | 0.864 | 0.309 |
| development | 27 | 4.538 | 0.781 | 0.340 |
| development | 28 | 5.003 | 0.816 | 0.419 |
| development | 29 | 5.593 | 0.821 | 0.467 |
| development | 30 | 6.773 | 0.844 | 0.594 |
| development | 31 | 9.272 | 0.802 | 0.802 |
| validation | 0 | 0.117 | 1.000 | -0.004 |
| validation | 1 | 0.185 | 1.000 | -0.006 |
| validation | 2 | 0.364 | 1.000 | -0.010 |
| validation | 3 | 0.461 | 0.912 | -0.006 |
| validation | 4 | 0.600 | 0.926 | 0.007 |
| validation | 5 | 0.703 | 0.942 | 0.005 |
| validation | 6 | 0.985 | 0.976 | 0.014 |
| validation | 7 | 1.020 | 0.955 | 0.013 |
| validation | 8 | 1.165 | 0.962 | 0.005 |
| validation | 9 | 1.229 | 0.974 | 0.006 |
| validation | 10 | 1.368 | 0.979 | 0.011 |
| validation | 11 | 1.378 | 0.963 | 0.006 |
| validation | 12 | 1.391 | 0.966 | -0.008 |
| validation | 13 | 1.511 | 0.972 | 0.016 |
| validation | 14 | 1.542 | 0.975 | 0.026 |
| validation | 15 | 1.602 | 0.952 | 0.032 |
| validation | 16 | 1.840 | 0.965 | 0.026 |
| validation | 17 | 1.975 | 0.971 | 0.058 |
| validation | 18 | 2.216 | 0.975 | 0.085 |
| validation | 19 | 2.476 | 0.904 | 0.130 |
| validation | 20 | 2.693 | 0.914 | 0.161 |
| validation | 21 | 2.977 | 0.918 | 0.203 |
| validation | 22 | 3.155 | 0.922 | 0.218 |
| validation | 23 | 3.488 | 0.866 | 0.240 |
| validation | 24 | 3.817 | 0.877 | 0.264 |
| validation | 25 | 3.941 | 0.883 | 0.282 |
| validation | 26 | 4.175 | 0.887 | 0.321 |
| validation | 27 | 4.678 | 0.821 | 0.353 |
| validation | 28 | 5.179 | 0.829 | 0.427 |
| validation | 29 | 5.739 | 0.836 | 0.494 |
| validation | 30 | 6.570 | 0.851 | 0.582 |
| validation | 31 | 8.016 | 0.807 | 0.807 |

Raw gate projections are not themselves gate activation values; sigmoid/softplus inside the kernel was not intercepted. Likewise the functional convolution output and delta-update term were not directly hookable with the existing module interface. Thus trace coverage is partial, and no first *causal* site follows from activation divergence. Machine source: `next_token_trace_*_v32.parquet`.
