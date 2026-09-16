# Architecture-resolved ceiling — V11

- Ceiling A: combined block-normalized raw dual-PCA, 599D.
- Ceiling B: separate REC + conv + KV coordinates, 1797D.
- Ceiling C: raw full persistent intervention, whose causal identity fidelity is 1 by definition.

| target | combined 599 | architecture 1797 | B − A | metric |
|---|---:|---:|---:|---|
| h1_j_effect | 0.530 | 0.642 | +0.112 | direction_cosine |
| h4_j_effect | 0.190 | 0.230 | +0.040 | direction_cosine |
| output_effect | 0.503 | 0.610 | +0.107 | correlation |

The old 599D result is therefore called a **combined-reference ceiling**, not a complete raw
persistent-state ceiling.

Machine records: `results/v11/processed/architecture_ceiling_v11.parquet` (`ef74609c1c412286a35643f199dc0c4bc03df72fdac54aaa4527f35b4cf7f41a`).
