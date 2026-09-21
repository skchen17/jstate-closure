# V21 practical action-coordinate ceiling

The six validation directions are distinct from the 12 training directions. All practical models use only positive train-action responses for fitting. No final-six response or raw final-action geometry was opened.

| fixed state/model | Z0 V20 descriptor | Z1 full requested | Z2 REC/Conv/KV corrected | Z3 realized BF16 Nyström |
|---|---:|---:|---:|---:|
| S2, G2 unseen direction relative L2 | 1.3137 | 0.6386 | 0.6329 | not separable for state-dependent Z3 |
| S2, G2 unseen sign relative L2 | 1.2479 | 0.9371 | 0.9125 | not separable |

Z3 was evaluated with the same frozen G4/G5 decoder class on S2 and S4. Best Z3 unseen-direction L2: **0.6983**; Z3 is a 39-dimensional train-anchor approximation to the actual BF16 REC/Conv/KV delta, not the full raw action. Best G4/G5 S2×rich-requested-Z unseen-direction L2: **0.7747**. G2 and G5 are not a one-axis comparison; fixed-G comparisons are in the factorial record.

Z1 retains the complete 14,397-dimensional frozen V13 score coordinate through an exact linear Gram. Z2 preserves three channel blocks. The first Z2 run blew up because the train KV median was effectively zero; its original result is retained, and a train-only robust normalization was frozen before the corrected run. Z0 has 16 dimensions. Full Z1 materially improves over Z0 under fixed S2/G2 by **0.6751** absolute L2, but **ACTION_COORDINATE_CEILING_PASS = False** because every practical candidate fails strict direction/sign gates; no action coordinate is declared sufficient.

Machine records: `action_representation_v21.json`, `bottleneck_factorial_v21.json`, `z2_channel_normalization_correction_v21.json`, `realized_action_factorial_v21.json`.
