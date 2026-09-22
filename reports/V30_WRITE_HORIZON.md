# Write Horizon — V30

A separate 10-state/role panel replays the same four prewrite-frozen tokens sequentially and compares h1/h2/h4; this is a single teacher-forced trajectory, not the four-probe concatenated primary endpoint.

| role | h | condition | cosine | magnitude | relative L2 |
|---|---|---|---|---|---|
| development | 1 | EXACT_REC_CONV | 0.965 | 0.977 | 0.269 |
| development | 1 | EXACT_CONV | 0.934 | 0.958 | 0.360 |
| development | 1 | GLOBAL_k32 | 0.824 | 0.861 | 0.576 |
| development | 1 | GLOBAL_k64 | 0.838 | 0.891 | 0.556 |
| development | 2 | EXACT_REC_CONV | 0.970 | 0.969 | 0.245 |
| development | 2 | EXACT_CONV | 0.938 | 0.909 | 0.372 |
| development | 2 | GLOBAL_k32 | 0.895 | 0.928 | 0.488 |
| development | 2 | GLOBAL_k64 | 0.899 | 0.876 | 0.491 |
| development | 4 | EXACT_REC_CONV | 0.925 | 0.879 | 0.383 |
| development | 4 | EXACT_CONV | 0.886 | 0.902 | 0.465 |
| development | 4 | GLOBAL_k32 | 0.749 | 0.818 | 0.669 |
| development | 4 | GLOBAL_k64 | 0.778 | 0.827 | 0.629 |
| validation | 1 | EXACT_REC_CONV | 0.974 | 1.002 | 0.227 |
| validation | 1 | EXACT_CONV | 0.948 | 0.960 | 0.335 |
| validation | 1 | GLOBAL_k32 | 0.856 | 0.888 | 0.535 |
| validation | 1 | GLOBAL_k64 | 0.877 | 0.926 | 0.500 |
| validation | 2 | EXACT_REC_CONV | 0.981 | 0.954 | 0.194 |
| validation | 2 | EXACT_CONV | 0.966 | 0.941 | 0.264 |
| validation | 2 | GLOBAL_k32 | 0.883 | 0.862 | 0.510 |
| validation | 2 | GLOBAL_k64 | 0.892 | 0.848 | 0.518 |
| validation | 4 | EXACT_REC_CONV | 0.959 | 0.922 | 0.285 |
| validation | 4 | EXACT_CONV | 0.932 | 0.926 | 0.368 |
| validation | 4 | GLOBAL_k32 | 0.747 | 0.802 | 0.692 |
| validation | 4 | GLOBAL_k64 | 0.770 | 0.827 | 0.641 |

Exact REC+Conv retains a donor-directed effect through h4 but development h4 does not meet the 0.30 L2 gate. Global compact writes do not regain qualification at later horizons. No transported-local h2/h4 claim is made because its h1 OOD gate failed.
