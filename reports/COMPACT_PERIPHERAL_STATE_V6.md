# Compact Peripheral State v6

## Material Passport

- Origin Skill: `academic-research-suite / experiment-agent`
- Mode: `gated compact-peripheral search`
- Date: `2026-09-15`
- Verification Status: `repository-grounded; machine results cited below`
- Protocol: `peripheral_foundations_protocol_v6`
- causal/null freeze digest: `f62c55d21e5e220cd21ea81c19a28bf1d5249a60bdc35a08e3eb029dd61ec12e`
- strong ceiling freeze digest: `6d4c77a3e33edf9e977e04293b73b77d153a15af091b1c496fe869e92ec23170`
- compact freeze digest: `07b51492b188b23da2ad66708cd3520e7b568a65bfd867ec019da7d301c24417`
- delivery/report freeze digest: `160d746a5b9b14e226acf876989e3196f971f321dbb9d2fb3720f7da68319840`


## Gate/result

The strong ceiling authorized the sweep. **1** of 18 candidates passed the joint screen. The smallest/only screen-pass was `pca 512D` with predictive gap closed `0.870642`, causal gap closed `0.997032`, causal-direction cosine `0.468382`, and semantic accuracy `0.615625`.

It did **not** pass conditional sufficiency: adding `residual(R|C)` changed next-J cosine by `0.001315`, causal-direction cosine by `0.009032`, and semantic accuracy by `0.022569`. The semantic gain is **2.26 percentage points**, above the frozen 2-point limit. These conditional values are exploratory single-seed point estimates; absence of a confirmatory CI is an additional reason not to authorize the compact state.

| family                |   dimension |   predictive_gap_closed |   causal_gap_closed |   action_accuracy |   causal_direction_cosine |   output_sign_agreement |
|:----------------------|------------:|------------------------:|--------------------:|------------------:|--------------------------:|------------------------:|
| pca                   |          16 |               -0.452829 |            0.57765  |          0.565625 |                  0.277039 |                0.575758 |
| pca                   |          32 |               -0.115247 |            0.729983 |          0.570486 |                  0.346541 |                0.454545 |
| pca                   |          64 |                0.259867 |            0.828826 |          0.574653 |                  0.391638 |                0.545455 |
| pca                   |         128 |                0.563321 |            0.912026 |          0.589583 |                  0.429598 |                0.575758 |
| pca                   |         256 |                0.751855 |            0.939749 |          0.601736 |                  0.442247 |                0.666667 |
| pca                   |         512 |                0.870642 |            0.997032 |          0.615625 |                  0.468382 |                0.545455 |
| predictive_bottleneck |          16 |                0.182403 |            0.579084 |          0.597917 |                  0.277693 |                0.575758 |
| predictive_bottleneck |          32 |                0.408723 |            0.755267 |          0.616667 |                  0.358076 |                0.575758 |
| predictive_bottleneck |          64 |                0.635979 |            0.912911 |          0.620486 |                  0.430002 |                0.545455 |
| predictive_bottleneck |         128 |                0.791924 |            0.948175 |          0.619444 |                  0.446091 |                0.484848 |
| predictive_bottleneck |         256 |                0.740478 |            0.916827 |          0.617708 |                  0.431788 |                0.515152 |
| predictive_bottleneck |         512 |                0.772082 |            0.89171  |          0.622569 |                  0.420329 |                0.515152 |
| nonlinear_bottleneck  |          16 |                0.360059 |            0.778795 |          0.603125 |                  0.368811 |                0.545455 |
| nonlinear_bottleneck  |          32 |                0.477614 |            0.835144 |          0.599306 |                  0.394521 |                0.545455 |
| nonlinear_bottleneck  |          64 |                0.662429 |            0.935307 |          0.611111 |                  0.44022  |                0.363636 |
| nonlinear_bottleneck  |         128 |                0.66123  |            0.940888 |          0.596181 |                  0.442766 |                0.484848 |
| nonlinear_bottleneck  |         256 |                0.749217 |            0.934023 |          0.612847 |                  0.439634 |                0.484848 |
| nonlinear_bottleneck  |         512 |                0.771475 |            0.962171 |          0.604167 |                  0.452477 |                0.515152 |

Figure: `results/v6/figures/compact_peripheral_pareto_v6.png`.

No recurrent controller was trained in v6.
