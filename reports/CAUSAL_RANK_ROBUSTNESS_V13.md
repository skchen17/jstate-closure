# Causal-rank robustness — V13

| probe family          |   median_r95 |   mean_r95 |
|:----------------------|-------------:|-----------:|
| architecture_balanced |        9.000 |      9.150 |
| causal_weighted       |        5.000 |      5.000 |
| high_variance_pca     |        7.500 |      7.900 |
| low_variance          |        8.500 |      8.200 |
| random_raw            |       11.000 |     11.000 |

Mean per-column sensitivities: `{'architecture_balanced': 4.660962577444547, 'causal_weighted': 2.3885091807515764, 'high_variance_pca': 0.7592895979669425, 'low_variance': 0.15593450524121333, 'random_raw': 0.7876877453097142}`. Low-variance/high-causal directions confirmed under the frozen rule: `False`. A single intrinsic dimension is not reported when construction-specific estimates disagree materially.
