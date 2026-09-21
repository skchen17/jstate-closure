# Pre-Readout Boundary Audit — V27

The accepted boundary is the prefix cache immediately before the final prompt token. REC, Conv, and KV fields already exist and are writable; the complete final-token forward, J/workspace extraction, final normalization, unembedding, and logits remain downstream. The persistent update has occurred for the prefix, but not for the final prompt token.

| target | replay p99 | normalized safety threshold |
|---|---:|---:|
| broad_vocabulary | 0 | 1e-12 |
| j | 0 | 1e-12 |
| late_residual | 0 | 1e-12 |
| logits | 0 | 1e-12 |
| semantic | 0 | 1e-12 |
| workspace | 0 | 1e-12 |

The ordinary reference perturbation changes current logits, semantic output, workspace, broad vocabulary, and late residual by median `0.865–1.085×` reference scale. Therefore current-output invariance is not structurally guaranteed at this boundary.
