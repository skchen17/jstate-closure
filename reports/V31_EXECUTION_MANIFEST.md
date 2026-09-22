# Execution Manifest — V31

Parent commit `5113e800cd4f287648cd52edc3577eb77cd60776`; base freeze `3f81577cb083ffac1d19b2e312fc7652417dd7f031186ef58b38763fe2f9a2e6`. Frozen panels 25/120/60/50; 192 token pairs 160/16/16; surface-composition triples 14/16/16; six future probes. TRAIN fits: 100 states, 1000 natural write rows, 1382 eligible composition examples, 200 exact REC+Conv response-factor examples. Development tested 40 AB compositions and 40 primitive contrasts; REC/Conv factorial and depth each used 20 pairs in development and 20 in validation. No independent final opened. A whole-GPU basis attempt hit concurrent GPU OOM; same frozen data were streamed by columns, with no gate/model change.

| freeze artifact | digest prefix | file hash prefix |
|---|---|---|
| compositional_natural_writes_v31.freeze.json | 3f81577cb083ffac | 159d2731367e5bda |
| compositional_natural_writes_v31_composition_development.freeze.json | f17dd13f29494469 | 3f475202a79bca91 |
| compositional_natural_writes_v31_composition_fit.freeze.json | 85f99e81a5efd04b | 2af8ca054e5f251a |
| compositional_natural_writes_v31_design.freeze.json | 8893b8b041256e18 | 023000b5f27fa587 |
| compositional_natural_writes_v31_execution_plan.freeze.json | 94cca9880d337796 | 83bd20dbd0aacd9d |
| compositional_natural_writes_v31_final_opening.freeze.json | d72f2baa448ebb21 | 6dbad3ddf3acd786 |
| compositional_natural_writes_v31_function_conditioned_fit.freeze.json | 9561baef8769b363 | 02a92a4ed8bfd9b7 |
| compositional_natural_writes_v31_mechanism_development.freeze.json | 20ef201e51007f96 | c995a6e262286214 |
| compositional_natural_writes_v31_mechanism_validation.freeze.json | 2203a5e37c11a4ce | 05f9b54621b68399 |
| compositional_natural_writes_v31_primitive_causal_development.freeze.json | 38045a86513e87ef | 22d3fc55fbbcd5af |
| compositional_natural_writes_v31_primitive_collect.freeze.json | 64b1ba2d6f3fdc28 | d273260f6fc9f288 |
| compositional_natural_writes_v31_primitive_dictionary_fit.freeze.json | e4ed869b7fdda47b | 645afede0678fa0b |
| compositional_natural_writes_v31_primitive_geometry.freeze.json | f1908e0ae1d5a61f | 7edae60266f54872 |
| compositional_natural_writes_v31_record_hash_manifest.freeze.json | 0f0a89615a76db58 | 7b448b5bed11bd9e |
| compositional_natural_writes_v31_response_factor_fit.freeze.json | 59309876e3ef1e25 | 1f9bfa27a78f7af8 |
| compositional_natural_writes_v31_train_depth_energy.freeze.json | 86985808887d3ca1 | 1bda0be362f25c2b |

Decision file `results/v31/processed/final_opening_v31.json` SHA-256 `44d7a75f098b27b57637fb19ecae176bf65ba445d72c845e490f31c4bb924616`. Authorization remains H2 true, H3/dynamic search/autonomous controller/cross-model false. Tests are recorded in `tests/test_v31_protocol.py`; off-repository scratch is not Git-tracked.
