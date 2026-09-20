# V20 finite operator rotation, gain and rank

Each state's positive shared-action response matrix has 18 rows × 288 normalized outputs. Its SVD yields r95; same-J P0/Pq pairs supply principal angles, gain, output-space Procrustes residual, spectrum divergence and rank change. This is finite-response geometry, not exact JVP geometry.

800 operator-state spectra and 600 same-base comparisons. Natural median r95=7.0000; counterfactual median r95=7.0000. Pair median principal angle=37.4441°, gain Pq/P0=1.0399, rank change=0.0000, Procrustes relative residual=0.6643. Channel-wise summaries: `{"conv": {"gain_ratio_Pq_over_P0": 0.8646912866833215, "mean_principal_angle_degrees": 38.08897258364372, "operator_modulation_relative": 0.8837564793419289, "procrustes_relative_residual": 0.7693639644014929, "rank_change": 0.0, "spectrum_Jensen_Shannon": 0.008270339641127082}, "joint": {"gain_ratio_Pq_over_P0": 1.0683720060770021, "mean_principal_angle_degrees": 30.363089012844426, "operator_modulation_relative": 0.6403150057559654, "procrustes_relative_residual": 0.492162263533666, "rank_change": 0.0, "spectrum_Jensen_Shannon": 0.005664421739958963}, "recurrent": {"gain_ratio_Pq_over_P0": 1.1645314276882308, "mean_principal_angle_degrees": 40.706882977544986, "operator_modulation_relative": 0.7929104062274832, "procrustes_relative_residual": 0.6923164661116596, "rank_change": -3.0, "spectrum_Jensen_Shannon": 0.03676571421902491}}`.

Historical V13 exact-JVP median instantaneous r95=8.0000, cumulative path r95=11.0000. Paired V20-state JVP subspaces were **not measured**; the historical ranks are context, not proof that tangent rotation and finite operator modulation are the same mechanism. Pair rows: `results/v20/processed/operator_rotation_pairs_v20.parquet`.

## Secondary family-rank aggregation

The median r95 is computed separately for natural and counterfactual operator states in each family (40 natural and 120 counterfactual states per family):

| family | natural median r95 | counterfactual median r95 |
|---|---:|---:|
| boolean_logic | 7 | 6 |
| modular_arithmetic | 7 | 7 |
| short_graph_traversal | 7 | 7 |
| simple_state_transition | 7 | 6.5 |
| variable_binding | 7 | 7 |

Median pairwise singular-spectrum Jensen–Shannon divergence is **0.0112**. These are descriptive ranks of the measured 18-action finite operator matrices. Source hashes and exact aggregations are in `results/v20/processed/operator_geometry_secondary_v20.json`.
