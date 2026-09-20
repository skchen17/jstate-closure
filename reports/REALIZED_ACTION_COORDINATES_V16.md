# Requested versus realized action coordinates — V16

The diagnostic reconstruction covers {'train': 20, 'validation': 10} and 3184 reliable actions. For each action, the exact BF16 state delta is projected by least squares onto the frozen 32-direction raw REC/Conv/KV span. Median out-of-span energy fraction (norm residual / realized norm): **0.010**. The response model comparison below refits and tests both inputs on the **same subset**; it is not compared unfairly to the 100/50 requested-input model.

| input                    |   k | model            |   train_count |   validation_count |   j_direction |   j_relative_l2 |   stacked_normalized_direction |   stacked_normalized_relative_l2 |
|:-------------------------|----:|:-----------------|--------------:|-------------------:|--------------:|----------------:|-------------------------------:|---------------------------------:|
| requested_raw_coordinate |  32 | channel_bilinear |          2111 |               1073 |         0.872 |           0.581 |                          0.849 |                            0.570 |
| realized_raw_coordinate  |  32 | quadratic        |          2111 |               1073 |         0.852 |           0.608 |                          0.841 |                            0.583 |

`requested_raw_coordinate = requested_alpha × z`; `realized_raw_coordinate` is the projected BF16 delta. The recorded residual prevents silently treating quantization spill outside the span as modeled action. Even if realized inputs improve prediction, this subset alone cannot establish quantization as the primary source of all V15 output nonlinearity. Records: `/data/CSK/J-space-project/jstate-closure/results/v16/processed/realized_action_coordinates_v16.parquet` and `/data/CSK/J-space-project/jstate-closure/results/v16/processed/requested_vs_realized_models_v16.parquet`.
