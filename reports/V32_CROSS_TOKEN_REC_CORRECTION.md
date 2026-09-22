# Cross-Token REC Correction — V32

The design froze two candidate contrasts per state, from different response-blind surface categories where eligible. Secondary factorial contrasts were evaluated in 25 development and 10 validation states, without fitting. The primary library also repeats token IDs across independent incoming states.

| role | distinct primary tokens | tokens in ≥2 states | max states/token | secondary contrasts | secondary median reduction | median within-state absolute difference |
|---|---|---|---|---|---|---|
| development | 38 | 29 | 6 | 25 | 0.418 | 0.208 |
| validation | 31 | 15 | 3 | 10 | 0.461 | 0.305 |

Within-state primary/secondary comparisons share probes and are directly comparable as donor-error reductions. Cross-state response-vector cosines are not, because each state's probe IDs differ. Category-stratified results appear in `V32_REC_CORRECTION_REPLICATION.md`. The observed variation is not a fitted token/state interaction law or REC content code. The full per-row records and token/state hashes are in `factorial_*_v32.parquet` and `design_v32.json`.
