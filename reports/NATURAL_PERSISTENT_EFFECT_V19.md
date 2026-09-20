# V19 natural persistent-state effects

N=Y(Pq,0)−Y(P0,0) is measured without probe a from two boundary-held-J states. N is not M; large N with small M would indicate a natural-dynamics/action-response dissociation only after the separate equivalence gate.

| h | median ‖N‖ stack | 95% state-bootstrap CI | median N/R0 |
|---|---|---|---|
| h1 | 6.2626 | [6.0956, 6.4186] | 0.8511 |
| h2 | 3.4762 | [3.1794, 3.7815] | 1.2848 |
| h4 | 1.8380 | [1.6169, 1.9254] | 1.4790 |
| h8 | 0.6865 | [0.5919, 0.8679] | 1.7526 |

## q-specific h1 effects

| q | states | median N/R0 | 95% CI | median M/R |
|---|---|---|---|---|
| conv_arch24_corrected | 100 | 2.1562 | [2.0176, 2.3105] | 1.0025 |
| conv_causal1 | 100 | 1.1585 | [1.1354, 1.1771] | 0.7227 |
| joint_causal11 | 100 | 0.8161 | [0.7729, 0.8529] | 0.5467 |
| joint_random2 | 100 | 0.0515 | [0.0479, 0.0550] | 0.0537 |
| kv_arch14_amended | 100 | 0.0223 | [0.0195, 0.0290] | 0.0312 |
| rec_arch4_amended | 100 | 1.3100 | [1.2463, 1.4056] | 0.7104 |
| rec_conv_causal1_amended | 100 | 1.6892 | [1.6062, 1.7562] | 1.0047 |
| rec_random7_weak_control | 100 | 0.0383 | [0.0364, 0.0398] | 0.0462 |

## Natural-effect direction coherence (h1 normalized stack)

| q | states | median ‖N‖ | median cosine to q mean N |
|---|---|---|---|
| conv_arch24_corrected | 100 | 14.6903 | 0.9106 |
| conv_causal1 | 100 | 7.6879 | 0.8701 |
| joint_causal11 | 100 | 5.2578 | 0.8434 |
| joint_random2 | 100 | 0.3469 | 0.7166 |
| kv_arch14_amended | 100 | 0.1421 | 0.0505 |
| rec_arch4_amended | 100 | 8.3770 | 0.9222 |
| rec_conv_causal1_amended | 100 | 11.1404 | 0.9170 |
| rec_random7_weak_control | 100 | 0.2477 | 0.6469 |

Train-frozen active q: conv_arch24_corrected, conv_causal1, joint_causal11, rec_arch4_amended, rec_conv_causal1_amended. Validation material-N gate: True. Secondary natural-dynamics outcome: **PERSISTENT_STATE_CAUSALLY_CONTRIBUTES_TO_NATURAL_CONTINUATION**. Weak-effect random/REC and KV directions remain explicit controls; a small M for an inactive q cannot support workspace sufficiency.
