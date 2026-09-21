# Input-Side Causal Geometry — V22

The 63 reliable V21 directions were corrected with the full 14,397-D raw-action Gram before decomposition. The Gram rank is 63, retained condition number is 5.99e+05, and the relative eigenvalue floor is `1e-8`.

| operator | input r90/r95/r99 | input P0→Pq angle | output P0→Pq angle |
|---|---:|---:|---:|
| exact JVP | [11.0, 15.0, 29.0] | 18.5340° | 18.2923° |
| finite | [14.0, 20.0, 34.0] | 31.4067° | 35.3716° |

JVP/finite input-subspace overlap is 0.5334 at P0 and 0.4576 at Pq. Input V rotates materially, so a fixed global action coordinate is structurally incomplete. Output rank is not used as a claim that the raw action space has the same dimension.
