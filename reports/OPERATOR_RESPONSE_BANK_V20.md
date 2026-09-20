# V20 crossed operator-response bank

For every persistent state P and shared frozen action a, `R_P(a)=Y(P,a)-Y(P,0)`. Y is the V16-normalized 288-D stack: J[0:128], logits[128:160], continuous semantic target[160:192], workspace[192:288]. Each base contributes natural P0 and three boundary-held-J Pq states, each crossed with the same **18 opened action directions** and both signs. The six final directions remain sealed. Train/validation/final action partitions: **12/6/6**, from **24 historically train-calibrated directions**; 32 was the target, 16 the minimum. The 24 retained directions meet the frozen historical selection and V20 train-only writeback checks. The third random control had 198/200 historical reliability; this was disclosed in an append-only amendment before operator responses.

Train: 150 bases, 600 operator states, 21600 response rows. Validation: 50 bases, 200 operator states, 7200 response rows. Direction tensors are referenced by frozen SHA/index/sign/alpha, not copied as multi-GB files. Every row stores requested/read-back actuator norms, cosine, gain, channel survival, no-action baseline, response and boundary-J identity.

| file | bases | operator states | rows | action reliable | SHA256 |
|---|---|---|---|---|---|
| results/v20/processed/response_operator_operator_train_boolean_logic_v20.parquet | 30 | 120 | 4320 | 1.0000 | 06f1cb2b465af6f9115efc8eb4ad79d5e823c9ac5ea349eb73f8c9fa8c9e24d2 |
| results/v20/processed/response_operator_operator_train_modular_arithmetic_v20.parquet | 30 | 120 | 4320 | 1.0000 | 21d3f8199c5e25424d65d44c605af1f20522b2b2edbb8f71fffe3b5d1c16c811 |
| results/v20/processed/response_operator_operator_train_short_graph_traversal_v20.parquet | 30 | 120 | 4320 | 1.0000 | c1842d06521f38e64c01a41356e4c0dcc73ec5157df3013a07480ea31e8fcc2b |
| results/v20/processed/response_operator_operator_train_simple_state_transition_v20.parquet | 30 | 120 | 4320 | 1.0000 | 1a11222aa0dbf501c8ac3feb3cb837641d0dcd49d69583131ba0d74498939a1e |
| results/v20/processed/response_operator_operator_train_variable_binding_v20.parquet | 30 | 120 | 4320 | 1.0000 | 498df5dbe6352a0b6f6e24454acca607389117ea982bf92e58e4ee30df765179 |
| results/v20/processed/response_operator_operator_validation_boolean_logic_v20.parquet | 10 | 40 | 1440 | 1.0000 | bd86a406a24e3541a6a7f8a4c63c5163d12f0d7ba70a849c45fb4fc91ae55196 |
| results/v20/processed/response_operator_operator_validation_modular_arithmetic_v20.parquet | 10 | 40 | 1440 | 1.0000 | 3522982173be195f58ec8c791ada5f5669765206d18fbc24c769e6514147d12c |
| results/v20/processed/response_operator_operator_validation_short_graph_traversal_v20.parquet | 10 | 40 | 1440 | 1.0000 | a1ac3faf6d90b5cdaa03d4aad653ed8d6ea6323be6b05b648b5f52e333d0d7d0 |
| results/v20/processed/response_operator_operator_validation_simple_state_transition_v20.parquet | 10 | 40 | 1440 | 1.0000 | 07e8b6043fedc167129fa8e51358d27301844322956ade940768d4a8ace99b46 |
| results/v20/processed/response_operator_operator_validation_variable_binding_v20.parquet | 10 | 40 | 1440 | 1.0000 | 2c24011e447cc6fc5616768f302db860852814cd0f6d2d2b6700106ac90772d0 |

Train IDs SHA `993e07d63d307105ae4d467b9f892a0466e23817326421a4be8e4fe0d84979fa`; validation IDs SHA `766ad5ccd0f6a16fde35990b649913af3f5b0818b0e56dbaf92f165ab35fb0bb`. Action hashes: `{"final_heldout": "cbfa6ce7c46aa4c400260c5a6fede1ea47999046e51d940f30db20e92b310aa2", "train": "af8cc82763286b93496d38a2401283e4fa9a5cb0e2c8524afb879d3f03bbb6c2", "validation": "9495edef649915c5bd58930f58f8d3960929661c1abcff20580d7af7686b8192"}`. Source design freeze `21fe1ee0f69b1559b00cd84cdd107a9f3b65809b419c99f7473b946024ace015`.
