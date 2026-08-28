# Phase 0 — J-lens validation

## Material Passport

- Run ID: `phase0-20260828T120038Z-cd33608d-s20260828`
- Verification Status: ANALYZED
- Gate: FAILED
- Failed criteria: `['hidden-intermediate hit@10']`
- Command: `/data/CSK/J-space-project/jstate-closure/src/jclosure/experiments/validate_lens.py --config configs/phase0_v1_replay.yaml`
- Model ID/revision: `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Lens revision/file SHA-256: `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a / 1f9a8f8fd593f0ffec1a9640993257ca4560f8ae3e5602315643d5cc6818534e`
- Upstream implementation commit: `581d398613e5602a5af361e1c34d3a92ea82ba8e`
- Model shard hashes: `{'model.safetensors-00001-of-00002.safetensors': {'size': 5329398688, 'sha256': '26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61'}, 'model.safetensors-00002-of-00002.safetensors': {'size': 3990429408, 'sha256': 'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188'}}`
- Vendored-data manifest SHA-256: `46a61ad9aaceaaea9b3d3ffcd52773c856cb4a39f5909ee4d92b71a01efe6af0`
- Full run manifest: `results/raw/phase0-20260828T120038Z-cd33608d-s20260828/manifest.json`

## Gate metrics

- Selected workspace band: `[22, 23, 24, 25, 26, 27, 28, 29, 30]`
- Hit@10 by family: `{'factual_two_hop': 0.03571428571428571, 'order_of_operations': 0.0}`
- Hit@10 by family/layer: `{'factual_two_hop': {'22': 0.03571428571428571, '23': 0.03571428571428571, '24': 0.07142857142857142, '25': 0.10714285714285714, '26': 0.14285714285714285, '27': 0.17857142857142858, '28': 0.17857142857142858, '29': 0.21428571428571427, '30': 0.2857142857142857}, 'order_of_operations': {'22': 0.5, '23': 0.5, '24': 0.5, '25': 0.5, '26': 0.5, '27': 0.5, '28': 0.0, '29': 0.0, '30': 0.5}}`
- Workspace-band threshold sensitivity: `{'0.25': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], '0.4': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], '0.6': [22, 23, 24, 25, 26, 27, 28, 29, 30]}`
- Rank-advantage CI: `{'estimate': 0.19540412223623568, 'lower': 0.10755061544833854, 'upper': 0.29815072717021995, 'confidence': 0.95, 'n_clusters': 25, 'n_observations': 30, 'n_resamples': 10000}`
- Positive-control CI: `{'estimate': 3.1283750863459203, 'lower': 2.1239655914471354, 'upper': 4.200688860482237, 'confidence': 0.95, 'n_clusters': 24, 'n_observations': 24, 'n_resamples': 10000}`
- Positive-control null threshold: `0.0001`

## Coverage and exclusions

- Raw layer/concept records: 13206
- Primary records: 2976
- First-16 exclusions: 9672
- Literal-copy exclusions: 496
- Positive-control records: 48
- Lens checkpoint `n_prompts`: 1000
- Companion fit-config `prompts_fitted`: 417
- The checkpoint/companion-metadata discrepancy is retained as a provenance warning.

If the gate failed, this is a measurement-system failure. Later causal
results must not be interpreted as evidence for H1, H2, or H3.
