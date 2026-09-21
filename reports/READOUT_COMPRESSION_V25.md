# Readout Compression — V25

| readout | centered r95 |
|---|---:|
| J_128 | 19 |
| direct_full_hidden_2560 | 47 |
| direct_residual_256 | 38 |
| logits_32 | 19 |
| random_late_hidden_1024 | 45 |
| random_late_hidden_128 | 35 |
| random_late_hidden_256 | 42 |
| random_late_hidden_512 | 44 |
| random_vocab_512 | 39 |
| semantic_32 | 15 |
| workspace_96 | 21 |

Direct full hidden r95 is `47` versus semantic-32 `15`, logits-32 `19`, J-128 `19`, and workspace-96 `21`. `READOUT_COMPRESSION_SUPPORTED = TRUE`, while the corrected internal convergence result is false; formal outcome `V25-E = TRUE`.
