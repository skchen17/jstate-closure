# Path dependence and holonomy — V14

Ten train-state loops A→B→C→A were constructed, with same-prompt A/B and C chosen by nearest frozen JVP matrix within family when available. This does not guarantee that C is close in full raw state or J-state distance. Mean results by mapping:

|   coordinate_return_error |   direct_indirect_coordinate_error |   j_effect_return_error | method             |   subspace_holonomy_degrees |
|--------------------------:|-----------------------------------:|------------------------:|:-------------------|----------------------------:|
|                     0.390 |                              0.390 |                   0.118 | grassmann_geodesic |                      22.587 |
|                     1.051 |                              1.062 |                   0.314 | j_response         |                      67.996 |
|                     0.390 |                              0.390 |                   0.118 | procrustes         |                      22.587 |
|                     0.501 |                              0.332 |                   0.181 | projector          |                      49.157 |
|                     0.000 |                              0.000 |                   0.000 | unaligned          |                       0.000 |

Procrustes/geodesic coordinate return error is `0.390`, with J-effect return error `0.118`. This is measurable path dependence in the tested local atlas, not proof that a low-dimensional manifold is absent. The zero error of `unaligned` is a tautology (identity map), not evidence of flat geometry. Degree-style holonomy is geometrically interpretable only for the orthogonal maps; projector and J-response maps are not isometries. Channel-wise return error in physical cache space was not measured, so that part of the requested audit remains unidentified.

Machine records: `results/v14/processed/holonomy_loops_v14.parquet`.
