# Read versus Write Causal Matrix — V28

Incoming/read interventions act before token t; outgoing/write interventions act only after token t has completed. Effects are V26 h1-reference-normalized Q.

| role | channel condition | read→Y_t | read→Y_t+1 | write→Y_t | write→Y_t+1 |
|---|---|---:|---:|---:|---:|
| development | Conv | 0.875 | 0.976 | 0.000 | 37.146 |
| development | KV_last_slot_previous_copy | 0.536 | 0.650 | 0.000 | 2.356 |
| development | REC | 0.814 | 0.909 | 0.000 | 13.690 |
| development | REC+Conv | 0.834 | 0.972 | 0.000 | 34.552 |
| development | REC+Conv+KV_last_slot_previous_copy | 0.832 | 0.977 | 0.000 | 35.359 |
| validation | Conv | 0.844 | 0.977 | 0.000 | 40.811 |
| validation | KV_last_slot_previous_copy | 0.555 | 0.668 | 0.000 | 2.354 |
| validation | REC | 0.765 | 0.969 | 0.000 | 13.601 |
| validation | REC+Conv | 0.878 | 0.965 | 0.000 | 36.818 |
| validation | REC+Conv+KV_last_slot_previous_copy | 0.858 | 0.954 | 0.000 | 37.353 |

The phase asymmetry is experimentally validated for the tested conditions, but the zero current write effect is guaranteed by the chosen after-readout interception time.
