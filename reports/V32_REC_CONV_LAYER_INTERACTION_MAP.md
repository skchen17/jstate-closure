# REC × Conv Layer Interaction Map — V32

The frozen four recurrent-layer groups and four Conv-layer groups form 16 exact-native pairings, each tested on 10 development and 10 validation states with six probes. The score is `(||donor−Conv_j||−||donor−REC_i+Conv_j||)/||donor−recipient||`. It is not an anatomical mediation score.

| role | REC group | Conv group | median donor-L2 improvement |
|---|---|---|---|
| development | early | early | 0.011 |
| development | early | early_mid | -0.002 |
| development | early | late | -0.005 |
| development | early | late_mid | -0.006 |
| development | early_mid | early | -0.004 |
| development | early_mid | early_mid | -0.009 |
| development | early_mid | late | -0.008 |
| development | early_mid | late_mid | -0.011 |
| development | late | early | 0.012 |
| development | late | early_mid | 0.009 |
| development | late | late | 0.013 |
| development | late | late_mid | 0.008 |
| development | late_mid | early | 0.001 |
| development | late_mid | early_mid | 0.001 |
| development | late_mid | late | -0.006 |
| development | late_mid | late_mid | -0.001 |
| validation | early | early | 0.005 |
| validation | early | early_mid | -0.002 |
| validation | early | late | -0.005 |
| validation | early | late_mid | 0.001 |
| validation | early_mid | early | -0.010 |
| validation | early_mid | early_mid | -0.003 |
| validation | early_mid | late | -0.010 |
| validation | early_mid | late_mid | -0.001 |
| validation | late | early | 0.008 |
| validation | late | early_mid | 0.007 |
| validation | late | late | 0.009 |
| validation | late | late_mid | 0.007 |
| validation | late_mid | early | 0.005 |
| validation | late_mid | early_mid | 0.005 |
| validation | late_mid | late | 0.002 |
| validation | late_mid | late_mid | -0.006 |

No individual group pair was selected from validation, and no single-layer refinement or formal pairwise gate was frozen. V32-H is therefore not established. Source: `rec_conv_layer_map_*_v32.parquet`.
