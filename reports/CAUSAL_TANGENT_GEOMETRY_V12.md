# Causal tangent geometry — V12

Principal angles are measured between right-singular subspaces in the same frozen 64-direction coordinate system.

|   grassmann_distance |   maximum_principal_angle_degrees |   mean_principal_angle_degrees |   rank | relation                        |   token_distance |   topk_overlap |
|---------------------:|----------------------------------:|-------------------------------:|-------:|:--------------------------------|-----------------:|---------------:|
|                1.812 |                            84.566 |                         38.544 |      8 | across_family                   |            0.500 |          0.589 |
|                2.793 |                            87.491 |                         43.924 |     16 | across_family                   |            0.500 |          0.511 |
|                3.506 |                            88.240 |                         35.854 |     32 | across_family                   |            0.500 |          0.615 |
|                1.755 |                            84.346 |                         36.533 |      8 | same_prompt_successive_position |            1.000 |          0.614 |
|                2.687 |                            87.277 |                         41.289 |     16 | same_prompt_successive_position |            1.000 |          0.548 |
|                3.431 |                            88.699 |                         34.655 |     32 | same_prompt_successive_position |            1.000 |          0.632 |

- Large tangent rotation (frozen ≥30° rule): `True`.
- Local rank reached the probe boundary: `False`.

`local rank low + angles large` 才支持 curved low-dimensional atlas；若 rank 接近 64-direction probe boundary，则只能报告 probe-limited evidence，不能声称全局低维。

Machine record: `results/v12/processed/causal_tangent_geometry_v12.parquet` (`5475e85d20d7ed1d8407ad474825a2700fef0d57d2c3375dd692cff20a3f681e`).
