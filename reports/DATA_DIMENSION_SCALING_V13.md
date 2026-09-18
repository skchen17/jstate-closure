# Data × dimension scaling — V13

| curve                           | N                  | h1 direction               | status       |
|:--------------------------------|:-------------------|:---------------------------|:-------------|
| architecture_resolved_pca/d1024 | 1200,2400,4800     | -0.135,-0.133,0.687        | DATA_LIMITED |
| architecture_resolved_pca/d128  | 600,1200,2400,4800 | -0.133,-0.174,-0.115,0.103 | DATA_LIMITED |
| architecture_resolved_pca/d256  | 600,1200,2400,4800 | -0.148,-0.058,-0.193,0.157 | DATA_LIMITED |
| architecture_resolved_pca/d512  | 600,1200,2400,4800 | -0.157,-0.106,-0.144,0.058 | DATA_LIMITED |
| global_causal_weighted/d1024    |                    |                            | DATA_LIMITED |
| global_causal_weighted/d128     | 600,1200,2400,4800 | -0.027,0.134,0.339,0.331   | DATA_LIMITED |
| global_causal_weighted/d256     | 600,1200,2400,4800 | -0.146,0.023,0.405,0.125   | DATA_LIMITED |
| global_causal_weighted/d512     | 600,1200,2400,4800 | 0.062,0.107,-0.029,0.149   | DATA_LIMITED |
| global_joint_pca/d1024          | 1200,2400,4800     | 0.510,0.081,0.635          | DATA_LIMITED |
| global_joint_pca/d128           | 600,1200,2400,4800 | -0.194,0.183,-0.130,0.192  | DATA_LIMITED |
| global_joint_pca/d256           | 600,1200,2400,4800 | 0.055,-0.141,-0.172,-0.038 | DATA_LIMITED |
| global_joint_pca/d512           | 600,1200,2400,4800 | 0.373,-0.136,-0.181,-0.011 | DATA_LIMITED |
| local_causal_basis/d1024        |                    |                            | DATA_LIMITED |
| local_causal_basis/d128         | 600,1200,2400,4800 | -0.027,0.134,0.348,0.749   | DATA_LIMITED |
| local_causal_basis/d256         | 600,1200,2400,4800 | -0.146,0.023,0.405,0.534   | DATA_LIMITED |
| local_causal_basis/d512         | 600,1200,2400,4800 | 0.062,0.107,-0.029,0.652   | DATA_LIMITED |

## Fixed 512D h1 metrics

| method                    |   train_size |   direction_h1 |   magnitude_h1 |   semantic_cosine_h1 |   output_direction_h1 |   semantic_sign_h1 |
|:--------------------------|-------------:|---------------:|---------------:|---------------------:|----------------------:|-------------------:|
| global_joint_pca          |          600 |          0.373 |          0.362 |                0.239 |                 0.068 |              0.300 |
| architecture_resolved_pca |          600 |         -0.157 |          0.357 |                0.147 |                 0.064 |              0.400 |
| global_causal_weighted    |          600 |          0.062 |          0.188 |                0.330 |                 0.018 |              0.300 |
| local_causal_basis        |          600 |          0.062 |          0.188 |                0.330 |                 0.018 |              0.300 |
| global_joint_pca          |         1200 |         -0.136 |          0.347 |                0.326 |                 0.063 |              0.400 |
| architecture_resolved_pca |         1200 |         -0.106 |          0.241 |                0.059 |                 0.015 |              0.400 |
| global_causal_weighted    |         1200 |          0.107 |          0.162 |                0.266 |                 0.066 |              0.400 |
| local_causal_basis        |         1200 |          0.107 |          0.162 |                0.266 |                 0.066 |              0.400 |
| global_joint_pca          |         2400 |         -0.181 |          0.500 |                0.309 |                 0.039 |              0.400 |
| architecture_resolved_pca |         2400 |         -0.144 |          0.494 |                0.262 |                 0.057 |              0.400 |
| global_causal_weighted    |         2400 |         -0.029 |          0.127 |               -0.325 |                 0.074 |              0.100 |
| local_causal_basis        |         2400 |         -0.029 |          0.127 |               -0.325 |                 0.074 |              0.100 |
| global_joint_pca          |         4800 |         -0.011 |          0.387 |                0.543 |                 0.037 |              0.400 |
| architecture_resolved_pca |         4800 |          0.058 |          0.331 |                0.394 |                 0.042 |              0.300 |
| global_causal_weighted    |         4800 |          0.149 |          0.203 |                0.376 |                 0.113 |              0.300 |
| local_causal_basis        |         4800 |          0.652 |          0.390 |                0.740 |                 0.125 |              0.600 |

The practical saturation rule was frozen at two consecutive doublings with absolute improvement below `0.01`. `NOT_IDENTIFIED_RANK_LIMIT` rows were retained in the parquet record; no rank was silently clamped. Local method: `family-stratified nearest-J neighborhood centered on one frozen validation representative per family; shared within family`. These curves are frozen held-out endpoint-fidelity estimates; finite writeback is adjudicated separately by the causal oracle.
