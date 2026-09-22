# Effect versus Tensor Primitives — V31

Tensor reconstruction and causal response were recorded on the same held-out rows.

| candidate | estimable rows | median tensor L2 | median causal L2 | Spearman across rows |
|---|---|---|---|---|
| PCA_M128 | 40 | 0.487 | 0.494 | 0.525 |
| SPARSE_DICTIONARY_M128_s16 | 40 | 0.513 | 0.518 | 0.631 |
| CLUSTERED_PROTOTYPES_M128_s16 | 40 | 0.524 | 0.525 | 0.557 |
| FUNCTION_CONDITIONED_M32 | 40 | 0.579 | 0.581 | 0.759 |
| RESPONSE_FACTOR_M128 | 40 | 0.559 | 0.565 | 0.301 |

Neither small tensor error nor positive rank correlation would by itself validate a primitive; the frozen donor-cosine/magnitude/L2/family gate is primary. These oracle projections are not held-out AB prediction.
