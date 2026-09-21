# V21 action-manifold coverage audit

| practical coordinate | median nearest train absolute cosine | median relative residual outside train span |
|---|---:|---:|
| Z0 | 0.6342 | 0.0992 |
| Z1 | 0.4181 | 0.7437 |
| Z2_corrected | 0.4213 | 0.7408 |
| Z3_actual_BF16 | 0.8801 | 0.0584 |

Z1/Z2 validation actions are far from the 12-direction train span, exceeding the frozen 0.25 residual coverage flag. Z0's small residual reflects its anchor-cosine descriptor, not true raw-action coverage. Z3's 39-dimensional projection similarly compresses raw BF16 geometry and cannot certify full raw support. Per-action values and REC/Conv/KV energy mismatch are in `results/v21/processed/action_coverage_audit_v21.json`.

Final six action responses and raw final-action geometry remain sealed; the requested final-coverage audit is withheld by the stronger independent-final rule, not marked passed.
