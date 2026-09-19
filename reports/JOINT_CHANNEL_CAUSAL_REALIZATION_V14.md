# Shared-coordinate channel realization — V14

The same local rank-3 coordinates were realized jointly in REC, convolution, and KV heads. Each channel was ablated at finite scale 1.0, without fitting a large decoder. Low-effect rows are retained but not classified as channel-required under the frozen `MIN_CAUSAL_EFFECT_NORM`.

- SNR-qualified ablation rows: `45`.
- Fraction marked `JOINT_CHANNEL_REQUIRED` when one-channel removal made J cosine <0.8 or norm ratio <0.8: `0.4222222222222222`.

|   j_cosine |   j_norm_ratio | removed_channel   |   required_fraction |
|-----------:|---------------:|:------------------|--------------------:|
|      0.386 |          0.328 | conv              |               1.000 |
|      1.000 |          0.999 | kv                |               0.000 |
|      0.840 |          0.976 | recurrent         |               0.267 |

This is a finite writeback ablation, not a learned shared latent decoder or complete-state replacement. Machine records: `results/v14/processed/joint_channel_ablation_v14.parquet`.
