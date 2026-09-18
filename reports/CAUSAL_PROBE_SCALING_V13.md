# Exact causal-probe scaling — V13

|       m |   median_r90 |   median_r95 |   median_r99 |   mean_stable_rank |   mean_effective_rank |
|--------:|-------------:|-------------:|-------------:|-------------------:|----------------------:|
|  64.000 |        4.000 |        5.000 |        8.000 |              2.460 |                 4.804 |
| 128.000 |        5.000 |        6.500 |       12.000 |              2.536 |                 5.496 |
| 256.000 |        5.000 |        8.000 |       16.500 |              2.582 |                 5.961 |
| 512.000 |        5.500 |        9.000 |       21.000 |              2.613 |                 6.321 |

All matrices use exact autograd JVP with Flash/memory-efficient SDP disabled. Ranks remain restricted to the frozen mixed empirical raw-state operator and are not full-state intrinsic dimensions.
