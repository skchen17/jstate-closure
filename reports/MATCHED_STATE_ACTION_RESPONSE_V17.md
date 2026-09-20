# V17 same-action, different-state pairs

Validation states were paired only where the V16 action has the same coordinate index, sign **and actual calibration alpha**; 57852 exact-action pairs were available. Raw and J distance were measured by frozen clean-state kernels. Type A/C are **rank-gap** top-decile diagnostics, not absolute closeness-matched pairs; family and prompt differences remain possible confounders.

| pair class | count | median J distance | median raw distance | median response relative divergence |
|---|---:|---:|---:|---:|
| A_similar_J_different_raw_rank_gap | 5788 | 1.5169 | 1.6443 | 0.8104 |
| C_similar_raw_different_J_rank_gap | 5808 | 1.6746 | 1.4883 | 0.9060 |
| other | 46256 | 1.4451 | 1.5100 | 0.7741 |

The relative response divergence is substantial but does not, by itself, prove J insufficiency under an exact J match. Type B (same J+C, different raw residual) is **undefined** because no compact C was selected. No context was physically swapped or injected. Full pair-level records: `results/v17/processed/matched_state_action_pairs_v17.parquet`.
