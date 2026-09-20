# V20 complete report

> This is the canonical single-file bundle for this version. It combines the 
> version-specific FINAL_REPORT section, every standalone version report, and 
> an integrity index of the machine-readable records. Standalone reports remain 
> preserved for direct navigation.

## Bundle provenance

- Source commit: `1b57e6f95238d8ae50603448ef426e48fa2b998b`
- Generated at: `2026-09-20T17:40:54.413438+00:00`
- Included standalone reports: `15`
- Indexed machine-record files: `36`
- Generator: `scripts/build_complete_version_report.py`

## Version summary and adjudication

## V20 — Compact Causal Response Operator State

Independent V19-B confirmation passed on **250** disjoint base states: same J, distinct P, matched finite action, M/R **0.7862** (95% state-bootstrap CI **[0.7782, 0.7931]**). At smaller reliable αq=0.25, all five selected active q retained nonzero descriptive modulation.

The crossed V20 response bank contains **150 training** and **50 validation** base states, each with natural P0 plus three same-J Pq states, crossed with **18** shared measured directions and both signs; **6** final directions remain sealed. The train-action oracle fingerprint is `Φ_train(P)=[R_P(a1),…,R_P(a12)]`, with 288-D normalized response per action. Five models × k=2–128 were tested; best validation diagnostic model `bilinear_latent_operator` at k=128 had unseen-direction stack relative L2 **0.9273**, unseen-sign **1.0056**. The full frozen direction/sign/scale/composition gate did **not** pass; `k_operator_min` is **not identified**.

Finite operator pair geometry: median principal angle **37.4441°**, gain Pq/P0 **1.0399**, r95 change **0.0000**. Operational same-J workspace-state aliasing observed: **True**. Historical V13 JVP rank is not a paired V20 subspace test.

Formal result: **V20-D_NO_COMPACT_OPERATOR_DIMENSION_IDENTIFIED**. Raw P→C encoder, conditional raw-gain equivalence, channel encoder audit and independent V20 final were **not eligible / unopened**, not passed or failed empirical tests. `V21_DYNAMIC_STATE_SEARCH_AUTHORIZED = FALSE`; H2 remains; H3 and autonomous training remain unauthorized. No physical cache replacement is licensed.

Protocol digest `8726bddd18d48665856ed35caefe4a32b25216ffddd578732b7f34565644069c`; action hashes `{"final_heldout": "cbfa6ce7c46aa4c400260c5a6fede1ea47999046e51d940f30db20e92b310aa2", "train": "af8cc82763286b93496d38a2401283e4fa9a5cb0e2c8524afb879d3f03bbb6c2", "validation": "9495edef649915c5bd58930f58f8d3960929661c1abcff20580d7af7686b8192"}`; V20 processed integrity index `results/v20/processed/v20_integrity_index.json`. See `reports/V20_COMPLETE_REPORT.md` for every standalone report in one file.

Append-only adjudication clarification: a separate frozen nonlinear k=128 model reached seen-direction/new-state relative L2 **0.2337** but failed unseen directions at **1.3318**. Thus **V20-C_ACTION_SPECIFIC_OPERATOR_ENCODING_ONLY** is a secondary descriptive finding, while the primary formal outcome remains V20-D. This correction is recorded in `results/v20/processed/v20_adjudication_amendment_1.json` with freeze digest `ed58a8a8314daa9bd6b2358b0a7565f48305a4a3440b66a8268a8d30d512dffc`; no data, model, threshold or gate was changed.

Test audit: 228 passed, 2 legacy cumulative-report hash tests failed (V14/V16); all six V20 tests passed. These old expectations already differed from the V19 parent commit, and the frozen old manifests were not modified. See `results/v20/processed/v20_test_audit.json`.

## Standalone report integrity index

| report | SHA256 |
|---|---|
| `reports/ACTION_COMPOSITION_GENERALIZATION_V20.md` | `12dc2abb9e9ccd2c8b29f752194c32ff0f440541009973f02bd4bf30134b7be5` |
| `reports/CHANNEL_OPERATOR_STATE_V20.md` | `9e897ae2514d0695141229d6f42b6a016ed22f94310c00c980927460bf5de968` |
| `reports/CONDITIONAL_RESPONSE_SUFFICIENCY_V20.md` | `e44582c52de17637f7eeb793ada7ba5b7728cd5c1dc9a885e940352de73686ff` |
| `reports/EXECUTION_MANIFEST_V20.md` | `bc0a4c50a1084b1812202cdb05c056e618e015729fe54066dfefd8a5aa56b44d` |
| `reports/NATURAL_VS_COUNTERFACTUAL_OPERATOR_V20.md` | `8c2833d0bd510bab4e519f192e45e597751e4aaa562f810285eccd8a79426e8f` |
| `reports/OPERATOR_INTRINSIC_DIMENSION_V20.md` | `6b43f752a2a2bf78fa7be56cc40a63d1bce3127c930444ae570eb94afa20917c` |
| `reports/OPERATOR_RESPONSE_BANK_V20.md` | `fbafd6355db11e0bc76012b86eb2161962a372224f8cdf7339f142f521d05f9b` |
| `reports/OPERATOR_ROTATION_GEOMETRY_V20.md` | `68e98009c7614ae54fa6c7c5f14a868fa97796010a454393d16029b7c18b8631` |
| `reports/ORACLE_OPERATOR_STATE_V20.md` | `52711429ea4342ca6162f1d79504c45fe32157f4b2dea425f1ca9e71d351711a` |
| `reports/RAW_TO_OPERATOR_ENCODER_V20.md` | `58c93d93b20beab9b27c5ba509dcfb4dca93787a59c0ecc662473d38466a339e` |
| `reports/STATE_ALIASING_V20.md` | `fe6e592d53024dbaaa0a7e7b805a325e5f0014b0296da197e4cfb9d09974104f` |
| `reports/STRICT_INTERFACE_AUDIT_V20.md` | `0fa962413748da9121693a666b57f88952e810e74a2f2b079bcf2636c36ffa56` |
| `reports/UNSEEN_ACTION_GENERALIZATION_V20.md` | `24f10697d445adac520367b887eb26f88b329823c0d75ac959114e944e23681d` |
| `reports/V19_INDEPENDENT_CONFIRMATION_V20.md` | `87bb0b80e9e44b0371ee581cb2734a12d28574257dc1f01634dde8a54014ab79` |
| `reports/V20_SCIENTIFIC_ANSWERS_V20.md` | `cb9b10035b761bd4e35a96e480c29e32df74e096fdce932e324ac201b322c780` |

## Machine-record integrity index

| record | bytes | SHA256 |
|---|---:|---|
| `results/v20/processed/action_calibration_v20.parquet` | 56641 | `f554c06f05b38e221de6b4455fb7ac498c8f4bcfeaeca609ea857085743702fa` |
| `results/v20/processed/action_descriptors_v20.json` | 10730 | `00311120a471c335335daac6d7ea5d23d491c187321a602d4a6bf2a51f453c97` |
| `results/v20/processed/independent_v19_confirmation_metrics_v20.parquet` | 938795 | `ad232b2569b3c0d7b9db313033d6508f6c1a8d01fa6b888bc124de53d86d6ab5` |
| `results/v20/processed/independent_v19_confirmation_v20.json` | 1813 | `908359c00c52d356afd8f53acac4568d18aee4e216f1e51f1311d10126bca2d5` |
| `results/v20/processed/independent_v19_confirmation_v20.parquet` | 24947400 | `db85777c95a1113774b280b21b4543b3fd53880b918b6c98c9d996746d25496b` |
| `results/v20/processed/operator_action_diagnostics_v20.json` | 5654 | `3306fdb6604838e1ed2ab8a93206d0ad780a353200b81f6ba76608b1a2b69054` |
| `results/v20/processed/operator_action_diagnostics_v20.parquet` | 291564 | `389f330ebc6bc9d48cda50101689f5239919ca5eacc8cda2607407b88ac6dfbe` |
| `results/v20/processed/operator_aliasing_pairs_v20.parquet` | 9291 | `20fd325b427ab5e793e548758b93f39213caac62fd3ded9f72265678c35fc005` |
| `results/v20/processed/operator_aliasing_v20.json` | 970 | `8792961cd9132e651c37f7d03786c592ab50ec39a2d571cc2144ef0627cf4e86` |
| `results/v20/processed/operator_geometry_secondary_v20.json` | 1677 | `739c8b8fea5eff1a8edf5b79389e7934817c018f2f8bc4044b92bf06d83dc2d6` |
| `results/v20/processed/operator_geometry_v20.json` | 3982 | `15e973e6466357b989f18ccd07373bf556a76b7684d6861e91102d753f49afe8` |
| `results/v20/processed/operator_rotation_pairs_v20.parquet` | 44930 | `0bd326c0faffad4188704aa2119b069ebbfe2acc7b21052d5372f184b7081d17` |
| `results/v20/processed/operator_singular_spectra_v20.parquet` | 172999 | `6f374861620e5c559a4637fc67d0fc260a571abeeabb8f32cb8c6d33c89bc112` |
| `results/v20/processed/operator_state_metadata_operator_train_v20.parquet` | 1227911 | `ad484b8f62220027a1be8aaddd11f30509a1882da7afa032ca36f9736558cd0d` |
| `results/v20/processed/operator_state_metadata_operator_validation_v20.parquet` | 441325 | `304f609f50e6ba8a38c140689f15ffc4d8790df771c68c5d2dbf3303a0470c24` |
| `results/v20/processed/oracle_operator_search_v20.json` | 111074 | `5e5356a6575a8f767548653811e591092c3908d312a76e2c3a54f80bb0033340` |
| `results/v20/processed/q_locality_calibration_v20.parquet` | 58364 | `893ead617c0d93ee0036e7b547f0b21a391b29ed34597a9231789f55de18647b` |
| `results/v20/processed/q_locality_v20.json` | 5298 | `b7ab3d69736cac846142979de82beb3bd322266d3a9f777e7a7f48b4200d1e41` |
| `results/v20/processed/q_locality_validation_v20.parquet` | 69138 | `082cd526489af63e8f01146043579034208f69bfcd9eab6a2c630799263f9d96` |
| `results/v20/processed/response_operator_operator_train_boolean_logic_v20.parquet` | 5022320 | `06f1cb2b465af6f9115efc8eb4ad79d5e823c9ac5ea349eb73f8c9fa8c9e24d2` |
| `results/v20/processed/response_operator_operator_train_modular_arithmetic_v20.parquet` | 4958893 | `21d3f8199c5e25424d65d44c605af1f20522b2b2edbb8f71fffe3b5d1c16c811` |
| `results/v20/processed/response_operator_operator_train_short_graph_traversal_v20.parquet` | 4998130 | `c1842d06521f38e64c01a41356e4c0dcc73ec5157df3013a07480ea31e8fcc2b` |
| `results/v20/processed/response_operator_operator_train_simple_state_transition_v20.parquet` | 4921405 | `1a11222aa0dbf501c8ac3feb3cb837641d0dcd49d69583131ba0d74498939a1e` |
| `results/v20/processed/response_operator_operator_train_v20.json` | 2562 | `b27764e1af50ffc1d371e1ea8e5a9817bef116f24a6f8c92b5ae225d187bf5ba` |
| `results/v20/processed/response_operator_operator_train_variable_binding_v20.parquet` | 4944363 | `498df5dbe6352a0b6f6e24454acca607389117ea982bf92e58e4ee30df765179` |
| `results/v20/processed/response_operator_operator_validation_boolean_logic_v20.parquet` | 1847928 | `bd86a406a24e3541a6a7f8a4c63c5163d12f0d7ba70a849c45fb4fc91ae55196` |
| `results/v20/processed/response_operator_operator_validation_modular_arithmetic_v20.parquet` | 1813417 | `3522982173be195f58ec8c791ada5f5669765206d18fbc24c769e6514147d12c` |
| `results/v20/processed/response_operator_operator_validation_short_graph_traversal_v20.parquet` | 1840752 | `a1ac3faf6d90b5cdaa03d4aad653ed8d6ea6323be6b05b648b5f52e333d0d7d0` |
| `results/v20/processed/response_operator_operator_validation_simple_state_transition_v20.parquet` | 1821871 | `07e8b6043fedc167129fa8e51358d27301844322956ade940768d4a8ace99b46` |
| `results/v20/processed/response_operator_operator_validation_v20.json` | 2590 | `c9e58a0e44327350652f951edf93973597e8a02a226bbac66a2163ed63a9d516` |
| `results/v20/processed/response_operator_operator_validation_variable_binding_v20.parquet` | 1826030 | `2c24011e447cc6fc5616768f302db860852814cd0f6d2d2b6700106ac90772d0` |
| `results/v20/processed/teacher_tokens_v20.parquet` | 29245 | `f0f4c86669f4695847d8990014cb4710f7670539186d87b347c19fd0212c449a` |
| `results/v20/processed/v20_adjudication.json` | 1540 | `4fc8826c909fac6c85c3547be3eee25b80c6b9a40053244949321a99cca14dcc` |
| `results/v20/processed/v20_adjudication_amendment_1.json` | 1533 | `639937b52c7cff3215bbc521b753c976a4063593ba3076fb796dd1650eb1bc15` |
| `results/v20/processed/v20_integrity_index.json` | 16838 | `abf2f578da1a1ca5cb028142351e1cb4e6375b9646833ee19760b3d66cfdb695` |
| `results/v20/processed/v20_test_audit.json` | 1406 | `1d312ae883336934d5186d6f40681b763c3e5d106598bbd4b8e001d4ce089a2c` |

---

## Bundled report 1: `ACTION_COMPOSITION_GENERALIZATION_V20.md`

# V20 unseen scale, pair and dense actions

The pre-frozen 10-base, 40-operator-state diagnostic panel measures 2× amplitude on one train and one validation direction, two unseen validation-direction pairs, and a five-direction dense mixture. The model sees each new continuous action descriptor but was never trained on the exact diagnostic combination. Action writeback and P0/Pq matching remain audited.

| action | type | stack relative L2 | median cosine | reliable rate | matched rate |
|---|---|---|---|---|---|
| unseen_dense_0_1_2_20_21 | unseen_dense | 0.8257 | 0.7127 | 1.0000 | 1.0000 |
| unseen_pair_20_16 | unseen_pair | 0.8487 | 0.7138 | 1.0000 | 1.0000 |
| unseen_pair_21_17 | unseen_pair | 1.0011 | 0.1153 | 1.0000 | 1.0000 |
| unseen_scale_train_0 | unseen_amplitude_train | 0.8354 | 0.9578 | 1.0000 | 1.0000 |
| unseen_scale_validation_20 | unseen_amplitude_validation | 0.8901 | 0.6671 | 1.0000 | 1.0000 |

Measured joint action is compared with the model's `G(C,a+b)`; additionally, additivity residual `||R(a+b)-R(a)-R(b)||/||R(a+b)||` is `{"unseen_pair_20_16": {"relative_additivity_residual_median": 0.28480471670627594}, "unseen_pair_21_17": {"relative_additivity_residual_median": 0.18583906441926956}}`. Additivity is not assumed. The joint sign/scale/composition gate is **False**. Raw diagnostics: `results/v20/processed/operator_action_diagnostics_v20.parquet`.

---

## Bundled report 2: `CHANNEL_OPERATOR_STATE_V20.md`

# V20 persistent-channel operator-state audit

**Not eligible / not executed:** the frozen oracle cross-action compactness gate failed. No raw-state encoder was trained and no result is imputed.

Equal-capacity REC, Conv, KV, pairwise and all-channel raw-P→C encoders were not trained; no channel-wise recoverability ranking is claimed. Descriptive q-channel finite-operator rotation/gain metrics are in `OPERATOR_ROTATION_GEOMETRY_V20.md` and are a different estimand.

---

## Bundled report 3: `CONDITIONAL_RESPONSE_SUFFICIENCY_V20.md`

# V20 conditional raw-response sufficiency

**Not eligible / not executed:** the frozen oracle cross-action compactness gate failed. No raw-state encoder was trained and no result is imputed.

`raw_incremental_gain = error(J,C,a) − error(J,C,P_raw,a)` was **not measured**. There is no equivalence CI or family-wise sufficiency result; the ≤0.02 practical margin cannot be invoked. H3 and physical replacement remain unauthorized.

---

## Bundled report 4: `EXECUTION_MANIFEST_V20.md`

# V20 execution and integrity manifest

Base protocol digest: `8726bddd18d48665856ed35caefe4a32b25216ffddd578732b7f34565644069c`; config SHA `4d1c61ac61f287005822726af39e8c59036d55ce690c06cbe2e71830690641a8`; parent Git commit `3cefc7c61429770eeef7fd6544adf4b9b17a1de5`. State role hashes: `{"independent_v19_confirmation": "596fee5c8fe0bf6afa31fd615731f115782750536b37a75daa07aff432e6fce8", "operator_train": "993e07d63d307105ae4d467b9f892a0466e23817326421a4be8e4fe0d84979fa", "operator_validation": "766ad5ccd0f6a16fde35990b649913af3f5b0818b0e56dbaf92f165ab35fb0bb", "q_calibration": "988c8ab70e93e9c86fcc7ebb6d78cf593cd155e1e46ad336f0a7fa87c47bc751", "reserved_independent_v20_final": "39fd933af48689767d5cbd574bf9f32b394bb41e1dd7345696aba6b1b3d398d1"}`. Action partition hashes: `{"final_heldout": "cbfa6ce7c46aa4c400260c5a6fede1ea47999046e51d940f30db20e92b310aa2", "train": "af8cc82763286b93496d38a2401283e4fa9a5cb0e2c8524afb879d3f03bbb6c2", "validation": "9495edef649915c5bd58930f58f8d3960929661c1abcff20580d7af7686b8192"}`. Operator q and action design digest: `21fe1ee0f69b1559b00cd84cdd107a9f3b65809b419c99f7473b946024ace015`. q-scale freeze `b80e50e7f8c2bdccb47173203e4c086dd911d3b325e0b61c73982bf271f0f37f`; diagnostic design `5543340e77d3b1cd2588d75eb93f6b42bff7e08cd1f6df12762b228225724261`; aliasing design `a4e54443016f65589b00413e1a7cb02a9e5dc3c247550b0df3c15ff3760dccac`.

Frozen manifests (14):
- `compact_causal_response_operator_v20.freeze.json` — `9ba5cbec830ce5a15fb75513b809aad10fc49552431cc9b9de12fb7626be7c23`
- `compact_causal_response_operator_v20_action_calibration.freeze.json` — `e210194ef09e6ca2cf4017a5ee9901925f9cad3c868c517958a879c7087a5f9a`
- `compact_causal_response_operator_v20_action_selection_amendment_1.freeze.json` — `21f0eae7f2a707107c9a84f013bae9cb23348de9ba5ac872b4cb969d7f5d240b`
- `compact_causal_response_operator_v20_actions.freeze.json` — `9ab92bf2d6feeff8fa2e78b2c9c21e4a679b1c76b18d6544f21d08ecce91c030`
- `compact_causal_response_operator_v20_geometry_design.freeze.json` — `5ffcca4036519665caa1102525e1f036fe2608ecf9a68e76ae8a749ded687be5`
- `compact_causal_response_operator_v20_independent_confirmation_design.freeze.json` — `edd1509db171bf0abb5c683cdb8cd012a06602a8521835d1d9963574c352cc8b`
- `compact_causal_response_operator_v20_operator_aliasing_design.freeze.json` — `9e0a3479d0eff8d0c5d773fa81deb995847daf9bfa0adefc9b7c403448491d97`
- `compact_causal_response_operator_v20_operator_analysis.freeze.json` — `d38b5f1a9431f7a6814090401b899cb48939b5d4f518588eb4d3990bbeab12f2`
- `compact_causal_response_operator_v20_operator_design.freeze.json` — `02d460be88a61cb8fca0aa020aa8e89298279a6c688ebd6abf0bfabd3f2300cc`
- `compact_causal_response_operator_v20_operator_diagnostics_design.freeze.json` — `ee92d3ee144ffc88b9c87b63ba721acfec750d3064ea6692a29c2ef848cf0e22`
- `compact_causal_response_operator_v20_q_locality_design.freeze.json` — `156fc8bd6369b806c143ff8a17fbe2b7d4f11314ee8d7e01088792176f8bb047`
- `compact_causal_response_operator_v20_q_locality_scales.freeze.json` — `2bf9b40d5d12db3e0e7bd731c91dcd7f8e3df99597bcd7572fec2c68c8e80cc8`
- `compact_causal_response_operator_v20_splits.freeze.json` — `fc51d5ad87f92bf85d1ac92832f9f7a4a30788fea60a024073011e8f43f8697c`
- `compact_causal_response_operator_v20_teacher.freeze.json` — `7b2a5790e5d49f8c4cbcd134ea624bf6a795fc036f4810b4d2b8cfb8b9c072ad`

Development response bank: 21600 train + 7200 validation rows. Independent V19-B bank: 16000 four-way rows. q-locality validation: 375 rows. Diagnostic action panel: 40 operator states. All output files and SHA256 values are indexed in `results/v20/processed/v20_integrity_index.json` after reports are built.

Executed module classes: `protocol_v20 freeze`; `operator_bank_v20 prepare`; `teacher_v20`; `independent_confirm_v20`; `actions_v20`; `q_locality_v20`; `response_operator_v20`; `operator_model_v20`; `operator_geometry_v20`; `operator_diagnostics_v20`; `operator_aliasing_v20`; `adjudicate_v20`. Sharded measurements ran on both GPUs. The frozen manifests and output records retain design hashes and completed counts.

Append-only post-adjudication clarification: `compact_causal_response_operator_v20_adjudication_amendment_1.freeze.json` adds a fifteenth freeze manifest (digest `ed58a8a8314daa9bd6b2358b0a7565f48305a4a3440b66a8268a8d30d512dffc`), linked to the original adjudication and model-sweep hashes. It identifies a secondary action-specific nonlinear fit without changing the primary V20-D outcome or opening final data. Derived family-rank aggregation is in `results/v20/processed/operator_geometry_secondary_v20.json`.

Test audit: `pytest -q` yielded **228 passed, 2 failed, 3 warnings**. All six V20 tests passed. Both failures are pre-existing cumulative-`FINAL_REPORT.md` hash expectations in V14/V16 frozen integrity tests; the V19 parent commit's cumulative report SHA (`a07031ac…`) already differs from V16's expected SHA (`8e19cf16…`). Historical manifests were preserved rather than rewritten. Details: `results/v20/processed/v20_test_audit.json`.

---

## Bundled report 5: `NATURAL_VS_COUNTERFACTUAL_OPERATOR_V20.md`

# V20 natural versus counterfactual operator manifolds

The distributions are separated: clean P0 states versus same-boundary-J Pq states. Centered equal-base action-response fingerprints yield r95 and entropy effective rank:

| role | distribution / q | states | r95 | entropy rank |
|---|---|---|---|---|
| operator_train | P0 | 150 | 39 | 11.8346 |
| operator_train | conv_causal1 | 150 | 44 | 11.9003 |
| operator_train | joint_causal11 | 150 | 39 | 13.0970 |
| operator_train | pooled_natural_plus_counterfactual | 600 | 50 | 16.0742 |
| operator_train | rec_arch4_amended | 150 | 35 | 11.6390 |
| operator_validation | P0 | 50 | 20 | 8.2223 |
| operator_validation | conv_causal1 | 50 | 23 | 9.1584 |
| operator_validation | joint_causal11 | 50 | 20 | 9.1144 |
| operator_validation | pooled_natural_plus_counterfactual | 200 | 34 | 14.0395 |
| operator_validation | rec_arch4_amended | 50 | 19 | 7.7261 |

This is an empirical rank on the frozen 18-direction local finite-action panel. It neither proves a global manifold dimension nor implies physical cache compression. Natural J→oracle-coordinate validation R²=0.1394; natural relative L2=0.9044; counterfactual relative L2=1.1608. The J predictor was fitted on natural training states only.

---

## Bundled report 6: `OPERATOR_INTRINSIC_DIMENSION_V20.md`

# V20 empirical operator dimension

The oracle coordinate is centered SVD of **positive train-action response fingerprints only**, fitted on training operator states. Held-out action responses never enter the coordinate. The frozen sweep tests k=2,4,8,16,32,64,128, subject to empirical rank, with five continuous-action decoders.

| k | train explained fraction | train fingerprint rel L2 | validation train-action rel L2 |
|---|---|---|---|
| 2 | 0.4867 | 0.5201 | 0.5290 |
| 4 | 0.6457 | 0.4321 | 0.4447 |
| 8 | 0.7829 | 0.3383 | 0.3547 |
| 16 | 0.8728 | 0.2589 | 0.2765 |
| 32 | 0.9288 | 0.1936 | 0.2168 |
| 64 | 0.9647 | 0.1364 | 0.1685 |
| 128 | 0.9860 | 0.0857 | 0.1266 |

Empirical train fingerprint rank is 600. This rank curve describes the 12-action inference fingerprint, not global dimension of P or of all possible actions. A low reconstruction error here alone is insufficient for cross-action compactness.

| model | k | seen rel L2 | unseen direction rel L2 | unseen sign rel L2 | all preliminary gates |
|---|---|---|---|---|---|
| response_svd | 2 | 0.8741 | 0.9358 | 1.0028 | False |
| reduced_rank_regression | 2 | 0.9054 | 0.9485 | 1.0170 | False |
| tucker_factorization | 2 | 0.9074 | 0.9446 | 1.0173 | False |
| bilinear_latent_operator | 2 | 0.8741 | 0.9358 | 1.0028 | False |
| small_nonlinear_latent_operator | 2 | 0.4741 | 1.7701 | 1.2508 | False |
| response_svd | 4 | 0.8584 | 0.9310 | 1.0047 | False |
| reduced_rank_regression | 4 | 0.8809 | 0.9315 | 1.0122 | False |
| tucker_factorization | 4 | 0.8828 | 0.9308 | 1.0148 | False |
| bilinear_latent_operator | 4 | 0.8584 | 0.9310 | 1.0047 | False |
| small_nonlinear_latent_operator | 4 | 0.3706 | 1.5973 | 1.2491 | False |
| response_svd | 8 | 0.8451 | 0.9305 | 1.0056 | False |
| reduced_rank_regression | 8 | 0.8571 | 0.9314 | 1.0094 | False |
| tucker_factorization | 8 | 0.8586 | 0.9302 | 1.0092 | False |
| bilinear_latent_operator | 8 | 0.8451 | 0.9305 | 1.0056 | False |
| small_nonlinear_latent_operator | 8 | 0.3175 | 1.4824 | 1.2032 | False |
| response_svd | 16 | 0.8351 | 0.9295 | 1.0060 | False |
| reduced_rank_regression | 16 | 0.8406 | 0.9306 | 1.0073 | False |
| tucker_factorization | 16 | 0.8412 | 0.9303 | 1.0071 | False |
| bilinear_latent_operator | 16 | 0.8351 | 0.9295 | 1.0060 | False |
| small_nonlinear_latent_operator | 16 | 0.2721 | 1.5075 | 1.1981 | False |
| response_svd | 32 | 0.8294 | 0.9285 | 1.0056 | False |
| reduced_rank_regression | 32 | 0.8310 | 0.9290 | 1.0060 | False |
| tucker_factorization | 32 | 0.8315 | 0.9290 | 1.0057 | False |
| bilinear_latent_operator | 32 | 0.8294 | 0.9285 | 1.0056 | False |
| small_nonlinear_latent_operator | 32 | 0.2356 | 1.5354 | 1.1751 | False |
| response_svd | 64 | 0.8259 | 0.9277 | 1.0057 | False |
| reduced_rank_regression | 64 | 0.8263 | 0.9277 | 1.0057 | False |
| tucker_factorization | 64 | 0.8264 | 0.9277 | 1.0057 | False |
| bilinear_latent_operator | 64 | 0.8259 | 0.9277 | 1.0057 | False |
| small_nonlinear_latent_operator | 64 | 0.2259 | 1.4317 | 1.1751 | False |
| response_svd | 128 | 0.8236 | 0.9273 | 1.0056 | False |
| reduced_rank_regression | 128 | 0.8237 | 0.9273 | 1.0057 | False |
| tucker_factorization | 128 | 0.8242 | 0.9274 | 1.0057 | False |
| bilinear_latent_operator | 128 | 0.8236 | 0.9273 | 1.0056 | False |
| small_nonlinear_latent_operator | 128 | 0.2337 | 1.3318 | 1.1587 | False |

Frozen gates: held-out J and full-stack median cosine ≥ 0.9, relative L2 ≤ 0.3, norm ratio in [0.8,1.2], each-family L2 ≤ 0.35, plus sign/scale/composition diagnostics. `k_operator_min` = **not identified**.

## Interpretation note

The reported numerical rank 600 uses a very small nonzero-singular-value tolerance and is **not** an intrinsic dimension estimate. The energy/reconstruction curve and cross-action held-out errors are the relevant empirical diagnostics. In particular, the nonlinear k=128 model attains seen-direction/new-state relative L2 **0.2337** while its unseen-direction L2 is **1.3318**: this is an action-specific fit, not a cross-action operator state. The poor generalization of this frozen action descriptor/model family cannot prove that no other compact operator parameterization exists.

---

## Bundled report 7: `OPERATOR_RESPONSE_BANK_V20.md`

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

---

## Bundled report 8: `OPERATOR_ROTATION_GEOMETRY_V20.md`

# V20 finite operator rotation, gain and rank

Each state's positive shared-action response matrix has 18 rows × 288 normalized outputs. Its SVD yields r95; same-J P0/Pq pairs supply principal angles, gain, output-space Procrustes residual, spectrum divergence and rank change. This is finite-response geometry, not exact JVP geometry.

800 operator-state spectra and 600 same-base comparisons. Natural median r95=7.0000; counterfactual median r95=7.0000. Pair median principal angle=37.4441°, gain Pq/P0=1.0399, rank change=0.0000, Procrustes relative residual=0.6643. Channel-wise summaries: `{"conv": {"gain_ratio_Pq_over_P0": 0.8646912866833215, "mean_principal_angle_degrees": 38.08897258364372, "operator_modulation_relative": 0.8837564793419289, "procrustes_relative_residual": 0.7693639644014929, "rank_change": 0.0, "spectrum_Jensen_Shannon": 0.008270339641127082}, "joint": {"gain_ratio_Pq_over_P0": 1.0683720060770021, "mean_principal_angle_degrees": 30.363089012844426, "operator_modulation_relative": 0.6403150057559654, "procrustes_relative_residual": 0.492162263533666, "rank_change": 0.0, "spectrum_Jensen_Shannon": 0.005664421739958963}, "recurrent": {"gain_ratio_Pq_over_P0": 1.1645314276882308, "mean_principal_angle_degrees": 40.706882977544986, "operator_modulation_relative": 0.7929104062274832, "procrustes_relative_residual": 0.6923164661116596, "rank_change": -3.0, "spectrum_Jensen_Shannon": 0.03676571421902491}}`.

Historical V13 exact-JVP median instantaneous r95=8.0000, cumulative path r95=11.0000. Paired V20-state JVP subspaces were **not measured**; the historical ranks are context, not proof that tangent rotation and finite operator modulation are the same mechanism. Pair rows: `results/v20/processed/operator_rotation_pairs_v20.parquet`.

## Secondary family-rank aggregation

The median r95 is computed separately for natural and counterfactual operator states in each family (40 natural and 120 counterfactual states per family):

| family | natural median r95 | counterfactual median r95 |
|---|---:|---:|
| boolean_logic | 7 | 6 |
| modular_arithmetic | 7 | 7 |
| short_graph_traversal | 7 | 7 |
| simple_state_transition | 7 | 6.5 |
| variable_binding | 7 | 7 |

Median pairwise singular-spectrum Jensen–Shannon divergence is **0.0112**. These are descriptive ranks of the measured 18-action finite operator matrices. Source hashes and exact aggregations are in `results/v20/processed/operator_geometry_secondary_v20.json`.

---

## Bundled report 9: `ORACLE_OPERATOR_STATE_V20.md`

# V20 oracle operator coordinate

`C_oracle(P)` is inferred from the same state's 12 positive train-action responses. Decoder fitting uses only train-state positive train-action responses and the continuous raw-direction descriptor (cosines to frozen train anchors plus channel norms). Validation directions, signs, amplitudes and action combinations do not supervise either coordinate or decoder. It is not an encoder from raw P.

Selected diagnostic model: `bilinear_latent_operator`, k=128; selected solely by validation unseen-direction stack relative L2. Seen-direction new-state L2=0.8236; unseen-direction L2=0.9273; unseen-sign train-direction L2=1.0056. Preliminary gate=False; full diagnostic gate=False. No compact operator-state is established; no inference about raw P compression follows.

Descriptor SHA `00311120a471c335335daac6d7ea5d23d491c187321a602d4a6bf2a51f453c97`; analysis freeze `c3f518b116a4bdfe54f7db38c3eddbc5c29519ea87b8a6881138a8f0fb7f5ed4`.

---

## Bundled report 10: `RAW_TO_OPERATOR_ENCODER_V20.md`

# V20 raw P to operator encoder

**Not eligible / not executed:** the frozen oracle cross-action compactness gate failed. No raw-state encoder was trained and no result is imputed.

Accordingly there is no measured `E(P)→C_oracle` result for REC, Conv, KV, full raw P, PLS, architecture-aware, interaction or nonlinear encoders. `C_oracle` remains response-derived and cannot be called a deployable compact state. Historical V19 raw-state encoder failure is not silently recycled as a V20 result.

---

## Bundled report 11: `STATE_ALIASING_V20.md`

# V20 workspace-state aliasing

Across 150 validation P0/Pq pairs, boundary J is exactly equal: **True**. Median standardized oracle-coordinate distance=12.3157; median train-action fingerprint difference=0.8083; fraction exceeding frozen coordinate and response thresholds=1.0000. Operational `WORKSPACE_STATE_ALIASING` observed: **True**.

The same J necessarily gives the same deterministic J-only prediction within a pair, while the measured train-action fingerprint can differ. This is an operational finite-action aliasing result, not a complete POMDP or Markov-state proof. Pair file `results/v20/processed/operator_aliasing_pairs_v20.parquet` (`20fd325b427ab5e793e548758b93f39213caac62fd3ded9f72265678c35fc005`).

---

## Bundled report 12: `STRICT_INTERFACE_AUDIT_V20.md`

# V20 strict inference and claim audit

1. The tested object is the local finite-action response operator `R_P(a)`, not physical P compression.
2. `C_oracle` sees only positive train-action response fingerprints from the same state. Held-out direction responses, negative signs, unseen scales, pairs, dense actions, teacher caches and final directions do not enter its inference or fit.
3. The response decoder sees continuous raw-direction-derived action features, not action-ID one-hot labels. Its five frozen model classes and k grid were set before validation responses.
4. The V19-B confirmation bank (250 bases) was disjoint from designated V18/V19 roles and completed before operator representation learning. Its positive result is independent of the V20 oracle failure.
5. All measured actuator failures remain in the denominator; no operator state was silently deleted. Train coordinate inference reliable states: 600/600; validation: 200/200.
6. Final held-out directions: `SEALED_UNOPENED`. Independent V20 final: `UNOPENED_NO_FROZEN_ELIGIBLE_ENCODER_FINALIST`. Neither is described as a passed test.
7. No compact causal response-state, dynamical Markov state, autonomous controller, cache replacement or H3 claim is authorized.

---

## Bundled report 13: `UNSEEN_ACTION_GENERALIZATION_V20.md`

# V20 unseen action generalization

Every validation operator state supplies its coordinate using **train actions only**. Frozen action partition is independent of state partition. Best diagnostic model metrics:

| test | stack relative L2 | J relative L2 | stack median cosine | J median cosine | stack median norm ratio |
|---|---|---|---|---|---|
| seen direction/new state | 0.8236 | 0.8408 | 0.8790 | 0.8907 | 0.1459 |
| unseen direction | 0.9273 | 1.0132 | 0.3828 | 0.2985 | 0.1919 |
| unseen sign/train direction | 1.0056 | 0.9264 | 0.5130 | 0.6701 | 0.1451 |
| unseen direction+sign | 1.0378 | 1.0726 | 0.1195 | -0.0144 | 0.1941 |

Family-wise unseen-direction relative L2: `{"boolean_logic": 0.914719198995127, "modular_arithmetic": 0.9512948995503161, "short_graph_traversal": 0.9278593231971451, "simple_state_transition": 0.9380887443778032, "variable_binding": 0.9215754106345686}`. A new state with a seen action is not evidence of action generalization. Final six directions were not opened because no encoder finalist was eligible.

---

## Bundled report 14: `V19_INDEPENDENT_CONFIRMATION_V20.md`

# V20 independent confirmation of V19-B

The new bank has **250 base states**, disjoint from V18/V19 designated roles. The boundary J is bitwise identical and persistent snapshots differ on every tested pair. This confirmation preceded V20 operator fitting.

Active q: median M/R **0.7862**, state-bootstrap 95% CI **[0.7782, 0.7931]**, 10000 matched rows; realized-action match 1.0000. Median N/R0 1.2420; N=8.6635, R0=6.9011, Rq=6.6857, M=5.9706. Response cosine=0.6493; ||Rq||/||R0||=0.9430.

| family | median M/R |
|---|---|
| boolean_logic | 0.7253 |
| modular_arithmetic | 0.7258 |
| short_graph_traversal | 0.8203 |
| simple_state_transition | 0.8400 |
| variable_binding | 0.8116 |

Formal result: **V19_B_INDEPENDENTLY_CONFIRMED**. This establishes persistent-state modulation of the tested finite responses, not compactness or dynamics. Raw rows: `results/v20/processed/independent_v19_confirmation_v20.parquet` (`db85777c95a1113774b280b21b4543b3fd53880b918b6c98c9d996746d25496b`).

---

## Bundled report 15: `V20_SCIENTIFIC_ANSWERS_V20.md`

# V20 answers to the 31 scientific questions

1. Yes: independent V19-B, 250 bases, M/R 0.7862 (95% CI 0.7782–0.7931).
2. Yes descriptively: at αq=0.25 all five selected q have reliable matched rate 1.0 and M/R 0.5198, 0.4186, 0.2147, 0.3456, 0.4686.
3. `Φ(P)` is the concatenation of 12 positive train-action 288-D response vectors; 18 directions × 2 signs were measured for diagnostics.
4. Train-positive fingerprint r95/energy curve is tabulated in OPERATOR_INTRINSIC_DIMENSION_V20.md; effective numerical rank 600.
5. Family-wise finite operator r95 is available in the singular-spectra Parquet; no common rank is assumed.
6. Natural and Pq empirical manifold dimensions are listed separately in NATURAL_VS_COUNTERFACTUAL_OPERATOR_V20.md.
7. No oracle coordinate passed all frozen cross-action gates, although train-action coordinates were constructed.
8. `k_operator_min` is not identified in k={2,4,8,16,32,64,128}.
9. Selected-model unseen-direction stack L2=0.9273; preliminary gate=False.
10. Unseen sign L2=1.0056; unseen direction+sign L2=1.0378.
11. Unseen amplitude results for train and validation directions are in ACTION_COMPOSITION_GENERALIZATION_V20.md.
12. Two unseen pair actions and one dense mixture were measured; metrics and actuator audits are in ACTION_COMPOSITION_GENERALIZATION_V20.md.
13. Same-J operator mean principal angle median=37.4441°.
14. Same-J median gain ratio Pq/P0=1.0399.
15. Same-J median r95 change=0.0000; full spectra and divergence are retained.
16. V13 exact-JVP historical rank is compared, but paired V20-state JVP subspaces were not measured.
17. Natural J→oracle C validation R²=0.1394.
18. Same J predicts identical C within each P0/Pq pair; counterfactual J→C relative L2=1.1608.
19. Operational workspace-state aliasing observed=True on 150 exact-J pairs.
20. Raw REC/Conv/KV→C was not eligible or trained after oracle gate failure.
21. No V20 channel-wise raw-P→C recoverability ranking was measured.
22. Same-J raw-P encoder response fidelity was not eligible or measured.
23. Raw P incremental information after J+C+a was not measured.
24. No practical-equivalence claim: raw gain and bootstrap CI were not measured.
25. Held-out family-wise oracle errors are reported; no all-family compact-state gate passed.
26. Independent V20 final status: UNOPENED_NO_FROZEN_ELIGIBLE_ENCODER_FINALIST.
27. No compact causal response-state candidate was established.
28. V21 dynamical-state search is not authorized.
29. H2 remains the project-level interpretation.
30. H3 is not authorized.
31. Autonomous state-model training is not authorized.

Formal outcome: **V20-D_NO_COMPACT_OPERATOR_DIMENSION_IDENTIFIED**. This is failure to identify compactness in the frozen tested model/action regime, not proof of mathematical nonexistence.

Append-only clarification for questions 4–8 and 25: the reported numerical rank 600 uses a near-zero singular-value tolerance and is not an intrinsic-dimension estimate. Family-wise measured 18-action finite-operator r95 medians are 7 on the natural states in all five families, and 6–7 on counterfactual states. A separate frozen nonlinear model at k=128 fits seen directions on new states (relative L2 0.2337) but fails unseen directions (1.3318), supporting **V20-C_ACTION_SPECIFIC_OPERATOR_ENCODING_ONLY** as a secondary descriptive finding. The primary V20-D gate result and V21/H3 restrictions do not change. Source: `results/v20/processed/v20_adjudication_amendment_1.json`.
