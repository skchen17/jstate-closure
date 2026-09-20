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
