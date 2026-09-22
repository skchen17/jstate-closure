# Multi-Probe Write Signature — V30

All main V30 causal rows concatenate J, selected logits, semantic log-probabilities, workspace, broad vocabulary and late residual across **four frozen next-token probes**. This is stricter than a single next token. Validation JOINT-OOD: exact REC+Conv cosine 0.976, global k64 0.828, local oracle k64 0.834. The four probes were fixed from prefix logits before current-token natural writes. This report does not claim semantic-coordinate labels.
