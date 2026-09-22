# Global versus Local Causal Models — V30

M0 fixed global, M1 family-fixed, M2 target local oracle, M3 source→target local transport are compared on **held-out target-state + held-out-token** cases. All local bases and transport maps use TOKEN_TRAIN only.

| role | model | n | cosine | magnitude | causal L2 | write L2 |
|---|---|---|---|---|---|---|
| development | GLOBAL_k32 | 60 | 0.791 | 0.848 | 0.619 | 0.596 |
| development | GLOBAL_k64 | 60 | 0.809 | 0.846 | 0.595 | 0.575 |
| development | FAMILY_k64 | 60 | 0.798 | 0.868 | 0.608 | 0.574 |
| development | LOCAL_ORACLE_k32 | 60 | 0.805 | 0.878 | 0.600 | 0.581 |
| development | LOCAL_ORACLE_k64 | 60 | 0.807 | 0.882 | 0.597 | 0.557 |
| development | TRANSPORTED_LOCAL_RIDGE_k32 | 60 | 0.801 | 0.878 | 0.600 | 0.582 |
| development | TRANSPORTED_LOCAL_RIDGE_k64 | 60 | 0.807 | 0.882 | 0.600 | 0.562 |
| development | TRANSPORTED_LOCAL_PROCRUSTES_k64 | 60 | 0.803 | 0.876 | 0.600 | 0.561 |
| validation | GLOBAL_k32 | 120 | 0.809 | 0.817 | 0.594 | 0.603 |
| validation | GLOBAL_k64 | 120 | 0.828 | 0.826 | 0.564 | 0.579 |
| validation | FAMILY_k64 | 120 | 0.829 | 0.830 | 0.565 | 0.571 |
| validation | LOCAL_ORACLE_k32 | 120 | 0.826 | 0.837 | 0.565 | 0.577 |
| validation | LOCAL_ORACLE_k64 | 120 | 0.834 | 0.845 | 0.556 | 0.558 |
| validation | TRANSPORTED_LOCAL_RIDGE_k32 | 120 | 0.823 | 0.836 | 0.571 | 0.582 |
| validation | TRANSPORTED_LOCAL_RIDGE_k64 | 120 | 0.830 | 0.841 | 0.562 | 0.559 |
| validation | TRANSPORTED_LOCAL_PROCRUSTES_k64 | 120 | 0.829 | 0.839 | 0.564 | 0.561 |

Local k64 and transported k64 are close, but both miss the strong causal ceiling. Consequently M0 is insufficient and M3 does not establish a shared command; M2 failure prevents attributing this solely to a bad transport map.
