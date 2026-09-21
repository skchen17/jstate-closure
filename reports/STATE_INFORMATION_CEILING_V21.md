# V21 state-information ceiling

The same Z1 requested-action representation and G2 decoder are held fixed throughout this table. S2/S3 are response-derived oracle contexts; S4 is a raw-state kernel approximation. None is a deployable learned compact state.

| state | feature dimension | seen/new state L2 | unseen direction L2 | unseen sign L2 | definition |
|---|---:|---:|---:|---:|---|
| S0 | 4096 | 0.6590 | 0.7565 | 0.9312 | full frozen current boundary J, 4,096-D; S0 dimension amendment |
| S1 | 128 | 0.1266 | 0.6409 | 0.9305 | V20 positive train-action fingerprint SVD k128 |
| S2 | 3456 | 0.0615 | 0.6386 | 0.9371 | uncompressed 12×288 positive train-action response fingerprint |
| S3 | 6912 | 0.0831 | 0.6393 | 0.9378 | uncompressed 24×288 signed train-action response fingerprint |
| S4 | 93 | 0.4605 | 0.6847 | 0.9410 | 93-D train-only, architecture-resolved raw-P Nyström reference; approximate raw-state kernel, not compact sufficiency |

The full S2 fingerprint improves on compact S1 by only **0.0023** absolute unseen-direction L2, below the frozen 0.10 materiality rule. S4 does not outperform S2 under the same Z1/G2. This does not prove a compact state exists or that raw P is useless: S4 is a 30-anchor Nyström approximation, while the 12 training directions leave large gaps in the requested action space. S0's base config label said 128, but the frozen V20 boundary-J record actually supplies 4,096 coordinates; the pre-fit append-only dimension amendment uses all 4,096.

State feature hashes: S0–S3 `33ceb530809ed31b4858d5a8a230f322f3a2de9b26d2b02c24ebad50d16142a7`; S4 `1b3fea502a21fba81149098e39210780257c6cc2a03b7c67d97b4940a5aec0be`. Validation unseen-action responses were never inputs to S0–S4.
