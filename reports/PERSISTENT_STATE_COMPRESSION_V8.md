# Persistent State Compression v8

Status: `COMPLETED_SCREEN_CAUSAL_GATED`.
Smallest authorized state: `None`.

The 16/32/64/128/256/512D sweep compares PCA/SVD, predictive and causal bottlenecks, nonlinear encoders, and layerwise fusion under universal, family-specific, and shared-plus-family-residual regimes.

## Pooled dimension sweep

| regime                 | method                |   dimension |   predictive_gap_closed |   conditional_residual_gain |   semantic_agreement | authorized   |
|:-----------------------|:----------------------|------------:|------------------------:|----------------------------:|---------------------:|:-------------|
| universal              | pca                   |          16 |             -0.00267821 |                  0.108448   |               0.2284 | False        |
| universal              | pca                   |          32 |              0.00514022 |                  0.107951   |               0.2272 | False        |
| universal              | pca                   |          64 |              0.0119602  |                  0.107082   |               0.2264 | False        |
| universal              | pca                   |         128 |              0.0349016  |                  0.104684   |               0.2284 | False        |
| universal              | pca                   |         256 |              0.203469   |                  0.0961863  |               0.2344 | False        |
| universal              | pca                   |         512 |              0.931401   |                  0.063026   |               0.2616 | False        |
| universal              | predictive_bottleneck |          16 |              0.074834   |                  0.105489   |               0.2284 | False        |
| universal              | predictive_bottleneck |          32 |              0.0986435  |                  0.102659   |               0.2316 | False        |
| universal              | predictive_bottleneck |          64 |              0.210611   |                  0.0956987  |               0.2392 | False        |
| universal              | predictive_bottleneck |         128 |              0.387533   |                  0.0852703  |               0.248  | False        |
| universal              | predictive_bottleneck |         256 |              0.707203   |                  0.0745653  |               0.2544 | False        |
| universal              | predictive_bottleneck |         512 |              0.939576   |                  0.0632537  |               0.264  | False        |
| universal              | causal_bottleneck     |          16 |              0.192221   |                  0.101431   |               0.2312 | False        |
| universal              | causal_bottleneck     |          32 |              0.309346   |                  0.0960824  |               0.2404 | False        |
| universal              | causal_bottleneck     |          64 |              0.471823   |                  0.0899698  |               0.2456 | False        |
| universal              | causal_bottleneck     |         128 |              0.713174   |                  0.0792785  |               0.2564 | False        |
| universal              | causal_bottleneck     |         256 |              0.869668   |                  0.0678939  |               0.2592 | False        |
| universal              | causal_bottleneck     |         512 |              0.964947   |                  0.0629723  |               0.2624 | False        |
| universal              | nonlinear_encoder     |          16 |              0.0193313  |                  0.107687   |               0.2292 | False        |
| universal              | nonlinear_encoder     |          32 |              0.0244081  |                  0.106597   |               0.2296 | False        |
| universal              | nonlinear_encoder     |          64 |              0.0911695  |                  0.103772   |               0.2312 | False        |
| universal              | nonlinear_encoder     |         128 |              0.148469   |                  0.10072    |               0.2284 | False        |
| universal              | nonlinear_encoder     |         256 |              0.20231    |                  0.0921742  |               0.2408 | False        |
| universal              | nonlinear_encoder     |         512 |              0.51552    |                  0.0803074  |               0.2476 | False        |
| universal              | layerwise_fusion      |          16 |             -0.00565018 |                  0.108411   |               0.2276 | False        |
| universal              | layerwise_fusion      |          32 |              0.00469434 |                  0.107991   |               0.2296 | False        |
| universal              | layerwise_fusion      |          64 |              0.0078607  |                  0.106937   |               0.228  | False        |
| universal              | layerwise_fusion      |         128 |              0.0382611  |                  0.104656   |               0.232  | False        |
| universal              | layerwise_fusion      |         256 |              0.135197   |                  0.0986625  |               0.2332 | False        |
| universal              | layerwise_fusion      |         512 |              0.894404   |                  0.0643459  |               0.264  | False        |
| family_specific        | pca                   |          16 |             -0.00667172 |                  0.106643   |               0.2292 | False        |
| family_specific        | pca                   |          32 |              0.0754908  |                  0.10414    |               0.2296 | False        |
| family_specific        | pca                   |          64 |              0.169858   |                  0.101667   |               0.2344 | False        |
| family_specific        | pca                   |         128 |              0.215046   |                  0.0986163  |               0.2332 | False        |
| family_specific        | pca                   |         256 |              0.209582   |                  0.0973877  |               0.2336 | False        |
| family_specific        | pca                   |         512 |              0.199079   |                  0.0950457  |               0.2336 | False        |
| family_specific        | predictive_bottleneck |          16 |              0.0694169  |                  0.10577    |               0.2316 | False        |
| family_specific        | predictive_bottleneck |          32 |              0.0992888  |                  0.104216   |               0.234  | False        |
| family_specific        | predictive_bottleneck |          64 |              0.140615   |                  0.101827   |               0.236  | False        |
| family_specific        | predictive_bottleneck |         128 |              0.213534   |                  0.0979373  |               0.238  | False        |
| family_specific        | predictive_bottleneck |         256 |              0.207536   |                  0.0967332  |               0.2376 | False        |
| family_specific        | predictive_bottleneck |         512 |              0.195227   |                  0.0944364  |               0.2372 | False        |
| family_specific        | causal_bottleneck     |          16 |              0.104911   |                  0.101942   |               0.2316 | False        |
| family_specific        | causal_bottleneck     |          32 |              0.153312   |                  0.100238   |               0.2348 | False        |
| family_specific        | causal_bottleneck     |          64 |              0.189115   |                  0.0983339  |               0.236  | False        |
| family_specific        | causal_bottleneck     |         128 |              0.211282   |                  0.0965058  |               0.236  | False        |
| family_specific        | causal_bottleneck     |         256 |              0.207199   |                  0.0953548  |               0.236  | False        |
| family_specific        | causal_bottleneck     |         512 |              0.200116   |                  0.0931546  |               0.2348 | False        |
| family_specific        | nonlinear_encoder     |          16 |              0.0254225  |                  0.106982   |               0.2272 | False        |
| family_specific        | nonlinear_encoder     |          32 |              0.0553519  |                  0.105773   |               0.2292 | False        |
| family_specific        | nonlinear_encoder     |          64 |              0.0703315  |                  0.103899   |               0.2296 | False        |
| family_specific        | nonlinear_encoder     |         128 |              0.183104   |                  0.0989545  |               0.232  | False        |
| family_specific        | nonlinear_encoder     |         256 |              0.17765    |                  0.0976932  |               0.232  | False        |
| family_specific        | nonlinear_encoder     |         512 |              0.16714    |                  0.0952943  |               0.2312 | False        |
| family_specific        | layerwise_fusion      |          16 |              0.0350592  |                  0.106944   |               0.2304 | False        |
| family_specific        | layerwise_fusion      |          32 |              0.0757987  |                  0.105194   |               0.2316 | False        |
| family_specific        | layerwise_fusion      |          64 |              0.127222   |                  0.102624   |               0.2324 | False        |
| family_specific        | layerwise_fusion      |         128 |          -3564.49       |                  0.00616532 |               0.1064 | False        |
| family_specific        | layerwise_fusion      |         256 |          -3517.23       |                  0.00611814 |               0.1064 | False        |
| family_specific        | layerwise_fusion      |         512 |          -3427.03       |                  0.00602591 |               0.1064 | False        |
| shared_family_residual | pca                   |          16 |              0.00657406 |                  0.108066   |               0.2288 | False        |
| shared_family_residual | pca                   |          32 |             -0.00837909 |                  0.106323   |               0.2292 | False        |
| shared_family_residual | pca                   |          64 |              0.0983097  |                  0.103364   |               0.23   | False        |
| shared_family_residual | pca                   |         128 |              0.179434   |                  0.100138   |               0.2336 | False        |
| shared_family_residual | pca                   |         256 |              0.242808   |                  0.0951516  |               0.2356 | False        |
| shared_family_residual | pca                   |         512 |              0.385384   |                  0.086694   |               0.2424 | False        |
| shared_family_residual | predictive_bottleneck |          16 |              0.0315128  |                  0.105189   |               0.2308 | False        |
| shared_family_residual | predictive_bottleneck |          32 |              0.142509   |                  0.102566   |               0.232  | False        |
| shared_family_residual | predictive_bottleneck |          64 |              0.204658   |                  0.0983353  |               0.2368 | False        |
| shared_family_residual | predictive_bottleneck |         128 |              0.341968   |                  0.0896795  |               0.246  | False        |
| shared_family_residual | predictive_bottleneck |         256 |              0.552444   |                  0.0772811  |               0.2548 | False        |
| shared_family_residual | predictive_bottleneck |         512 |              0.836795   |                  0.0676053  |               0.2592 | False        |
| shared_family_residual | causal_bottleneck     |          16 |              0.177477   |                  0.0992069  |               0.2352 | False        |
| shared_family_residual | causal_bottleneck     |          32 |              0.353618   |                  0.0949669  |               0.2372 | False        |
| shared_family_residual | causal_bottleneck     |          64 |              0.495135   |                  0.0884211  |               0.2468 | False        |
| shared_family_residual | causal_bottleneck     |         128 |              0.671561   |                  0.0811525  |               0.252  | False        |
| shared_family_residual | causal_bottleneck     |         256 |              0.89575    |                  0.0706491  |               0.2568 | False        |
| shared_family_residual | causal_bottleneck     |         512 |              1.01423    |                  0.0607412  |               0.2668 | False        |
| shared_family_residual | nonlinear_encoder     |          16 |              0.0276512  |                  0.107269   |               0.2288 | False        |
| shared_family_residual | nonlinear_encoder     |          32 |              0.0440895  |                  0.106294   |               0.2284 | False        |
| shared_family_residual | nonlinear_encoder     |          64 |              0.0834333  |                  0.104267   |               0.2292 | False        |
| shared_family_residual | nonlinear_encoder     |         128 |              0.180584   |                  0.0991902  |               0.2348 | False        |
| shared_family_residual | nonlinear_encoder     |         256 |              0.316047   |                  0.0918869  |               0.2344 | False        |
| shared_family_residual | nonlinear_encoder     |         512 |              0.386361   |                  0.0835861  |               0.2432 | False        |
| shared_family_residual | layerwise_fusion      |          16 |              0.0116355  |                  0.107897   |               0.2312 | False        |
| shared_family_residual | layerwise_fusion      |          32 |              0.0372699  |                  0.106589   |               0.2312 | False        |
| shared_family_residual | layerwise_fusion      |          64 |              0.0794953  |                  0.104479   |               0.2312 | False        |
| shared_family_residual | layerwise_fusion      |         128 |              0.138719   |                  0.100985   |               0.234  | False        |
| shared_family_residual | layerwise_fusion      |         256 |          -3552.56       |                  0.00604611 |               0.1064 | False        |
| shared_family_residual | layerwise_fusion      |         512 |          -3461.14       |                  0.00598227 |               0.1068 | False        |

## Family-wise sensitivity

| family                  |   dimension |   best_predictive_gap |   minimum_conditional_gain | any_authorized   |
|:------------------------|------------:|----------------------:|---------------------------:|:-----------------|
| boolean_logic           |          16 |           0.188374    |                 -0.016897  | False            |
| boolean_logic           |          32 |           0.309672    |                 -0.0164199 | False            |
| boolean_logic           |          64 |           0.429464    |                 -0.0163452 | False            |
| boolean_logic           |         128 |           0.564401    |                 -0.015192  | False            |
| boolean_logic           |         256 |           0.611493    |                 -0.0159918 | False            |
| boolean_logic           |         512 |           0.991923    |                 -0.0159545 | False            |
| modular_arithmetic      |          16 |           2.83215e+09 |                 -0.0184338 | False            |
| modular_arithmetic      |          32 |           1.15626e+10 |                 -0.020409  | False            |
| modular_arithmetic      |          64 |           1.37205e+10 |                 -0.0206499 | False            |
| modular_arithmetic      |         128 |           1.16628e+10 |                 -0.0205392 | False            |
| modular_arithmetic      |         256 |           1.58603e+10 |                 -0.0202719 | False            |
| modular_arithmetic      |         512 |           1.42227e+10 |                 -0.018354  | False            |
| short_graph_traversal   |          16 |           0.340172    |                 -0.0469589 | False            |
| short_graph_traversal   |          32 |           2.73308e+09 |                 -0.0448523 | False            |
| short_graph_traversal   |          64 |           9.13977e+09 |                 -0.0424963 | False            |
| short_graph_traversal   |         128 |           0.716916    |                 -0.0425953 | False            |
| short_graph_traversal   |         256 |           0.912695    |                 -0.0413141 | False            |
| short_graph_traversal   |         512 |           1.13512     |                 -0.0401797 | False            |
| simple_state_transition |          16 |           0.300804    |                 -0.0180197 | False            |
| simple_state_transition |          32 |           0.385871    |                 -0.0177568 | False            |
| simple_state_transition |          64 |           0.528196    |                 -0.0184891 | False            |
| simple_state_transition |         128 |           0.72166     |                 -0.0185446 | False            |
| simple_state_transition |         256 |           0.76091     |                 -0.0190037 | False            |
| simple_state_transition |         512 |           0.953684    |                 -0.0187771 | False            |
| variable_binding        |          16 |           1.53641e+10 |                 -0.0357082 | False            |
| variable_binding        |          32 |           1.49691e+10 |                 -0.0362506 | False            |
| variable_binding        |          64 |           1.39299e+10 |                 -0.0365412 | False            |
| variable_binding        |         128 |           1.43678e+10 |                 -0.0361678 | False            |
| variable_binding        |         256 |           1.54338e+10 |                 -0.0355218 | False            |
| variable_binding        |         512 |           3.97915e+07 |                 -0.0358239 | False            |
