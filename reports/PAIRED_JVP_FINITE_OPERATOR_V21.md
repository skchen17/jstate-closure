# V21 paired exact differential and finite operator panel

Exactly matched P0/Pq persistent states were measured on **50 development** and **25 disjoint validation** bases, balanced across five families. All matrices use 64 frozen probes; one train-calibrated BF16-unreliable probe is retained in raw audit but excluded from reliability-qualified 63-column spectra. First 18 probes are the V20 12 train plus 6 validation actions; none of the final six responses was opened. The same frozen h1 288-D target and teacher token are used for exact JVP and signed central finite response.

| role | bases | JVP median r90/r95/r99 | finite median r90/r95/r99 | JVP P0/Pq median angle | finite P0/Pq median angle |
|---|---:|---|---|---:|---:|
| development | 50 | 4.0 / 5.0 / 8.0 | 5.0 / 7.0 / 12.0 | 36.9025° | 29.0307° |
| validation | 25 | 4.0 / 5.0 / 8.0 | 5.0 / 7.0 / 12.0 | 36.7976° | 30.4461° |

Exact JVP uses forward-mode `torch.func.jvp` for all 75 final-panel bases. The original reverse-over-reverse 17-base development run is preserved separately, never mixed. Five-probe same-state comparison found minimum column cosine 0.999953 and maximum relative difference below 0.01; see `forward_ad_comparison_v21.json` and the append-only method amendment. Raw matrices remain outside Git; `paired_jvp_finite_operator_v21.parquet` indexes their hashes.
