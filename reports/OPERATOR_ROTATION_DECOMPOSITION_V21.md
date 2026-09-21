# V21 rotation versus gain/spectrum/rank decomposition

For each identical P0/Pq pair the operator change was decomposed via output-space orthogonal Procrustes, gain and singular spectra. The original draft Procrustes formula mistakenly used `nuclear_norm(AᵀB)`; a pre-analysis CPU invariant caught this. The append-only correction uses the equivalent compact SVD of `B Aᵀ`, with no raw matrix change.

| validation finite P0→Pq statistic | median |
|---|---:|
| pre-alignment relative error | 0.7697 |
| post-orthogonal-Procrustes relative error | 0.4033 |
| Pq/P0 Frobenius gain | 0.9583 |
| singular-spectrum JS divergence | 0.006929 |
| r95 difference | 0.0 |
| JVP P0/Pq median principal angle | 36.7976° |
| finite P0/Pq median principal angle | 30.4461° |

An error reduction after Procrustes is compatible with rotation, but not by itself causal proof that one operator is exactly a rotated copy of the other. Family-split association and JVP–finite overlap determine the stronger shared-mechanism claim.
