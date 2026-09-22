# Primitive Causal Realization — V31

All 40 development held-out AB/state pairs used the exact V29/V30 native writeback interface and six fixed future probes. Dictionary coefficients are oracle projections of the *held-out natural write* onto TRAIN-only atoms: this is a compression/reconstruction ceiling, **not** prediction of AB from A/B. Reconstruction tensor L2 is secondary; future-response cosine, magnitude and relative L2 decide.

| condition | rows | median cosine | median magnitude | median causal L2 | gate pass |
|---|---|---|---|---|---|
| EXACT_REC_CONV | 40 | 0.967 | 0.975 | 0.255 | False |
| PCA_M32 | 40 | 0.839 | 0.878 | 0.547 | False |
| PCA_M64 | 40 | 0.856 | 0.872 | 0.519 | False |
| PCA_M128 | 40 | 0.872 | 0.899 | 0.494 | False |
| SPARSE_DICTIONARY_M128_s4 | 40 | 0.831 | 0.876 | 0.562 | False |
| SPARSE_DICTIONARY_M128_s16 | 40 | 0.859 | 0.896 | 0.518 | False |
| CLUSTERED_PROTOTYPES_M128_s4 | 40 | 0.826 | 0.852 | 0.566 | False |
| FUNCTION_CONDITIONED_M32 | 40 | 0.817 | 0.869 | 0.581 | False |
| RESPONSE_FACTOR_M128 | 40 | 0.826 | 0.856 | 0.565 | False |

Passing non-ceiling development candidates: `[]`. The full machine table contains every estimable M/s condition and unavailable rows. Per-family ≥4/5 and per-row gates were applied without retuning. Because no finalist qualified, the frozen primitive validation/final panels stayed sealed.
