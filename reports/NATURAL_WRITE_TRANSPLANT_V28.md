# Natural Write Transplant — V28

Two independently computed natural branches per prompt were formed by a frozen incoming perturbation. The recipient current computation stayed untouched; donor outgoing native fields were inserted before the same next token. Full REC+Conv+KV replacement passed both directions in development and validation: median cosine ≈1, magnitude ratio 1, relative L2 to donor 0.

This complete-cache result is expected when the entire same-length future cache is made identical to the donor and the same token is replayed. It confirms fidelity and full-state transferability, **not** selective channel mediation or a compact state. REC+Conv partial transplant failed (validation A←B median cosine `0.500508`; B←A `0.566383`). Single-channel transplants also failed the reciprocal directional gate.
