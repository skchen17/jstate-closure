# V17 immediate target audit

All h1 targets in the V16 action bank were evaluated under the **same train-selected ridge and state split**; 35 target×context metrics are in `h1_all_target_metrics_v17.json/parquet`. Relative L2, median response direction, median magnitude ratio and component-sign agreement are machine-readable.

| target | J-only rel L2 | J+all rel L2 | full raw rel L2 | J-only direction | J+all direction |
|---|---:|---:|---:|---:|---:|
| j | 0.2989 | 0.2863 | 0.2800 | 0.9601 | 0.9627 |
| logits | 0.3471 | 0.3338 | 0.3287 | 0.9485 | 0.9522 |
| semantic_continuous | 0.3668 | 0.3569 | 0.3555 | 0.9530 | 0.9555 |
| workspace | 0.3464 | 0.3356 | 0.3340 | 0.9419 | 0.9449 |
| stacked_normalized | 0.3518 | 0.3406 | 0.3377 | 0.9505 | 0.9539 |

The V16 finite-action bank stores continuous semantic response, not the historical discrete legacy semantic score. That legacy metric is explicitly **unavailable**, not treated as zero or silently replaced. The normalized stack combines targets under V16 frozen scales.
