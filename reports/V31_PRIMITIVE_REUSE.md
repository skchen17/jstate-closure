# Primitive Reuse — V31

For M128/s4 oracle encodings, reuse is counted only when an atom appears with ≥2 token IDs, ≥2 states and ≥2 task families.

| candidate | used atoms | geometric multi-token/state/family atoms | max families/atom | causal gate |
|---|---|---|---|---|
| SPARSE_DICTIONARY | 50 | 15 | 5 | False |
| CLUSTERED_PROTOTYPES | 53 | 12 | 5 | False |

A repeated OMP index is **geometric reuse only**. Since the held-out causal gate fails, no index is promoted to a reusable causal primitive. No semantic label is assigned to PCA axes or clusters.
