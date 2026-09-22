# V33 Dimensionless Cross-Model Effect Shapes

|model / role|REC-only L2|Conv-only L2|joint L2|correction reduction|alignment|interaction ratio|
|---|---:|---:|---:|---:|---:|---:|
|Qwen3.5 / development|0.998|0.283|0.163|0.410|0.809|0.252|
|Qwen3.5 / validation|0.997|0.288|0.180|0.372|0.779|0.247|
|Falcon-H1 / development|0.997|0.242|0.135|0.388|0.793|0.224|
|Falcon-H1 / validation|1.001|0.240|0.126|0.413|0.811|0.228|
|Falcon-H1 / final|0.997|0.227|0.140|0.363|0.776|0.219|

All values are within-model dimensionless normalizations; raw hidden/logit norms are not compared across models. The structural ordering and conditional alignment replicate closely, but two models cannot establish universality. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.
