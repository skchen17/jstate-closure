# V33 Execution Manifest

Parent `09a00a67e5a286a05e47730c43586bce94b8095a`; base `aa843d414c7ad5866a47beebc665db9d0379d2a622215b1465cdc4d6ae41b600`. Fixed Falcon revision `06d7330266253c20784f58b0a846dc09a9d12cee`; bfloat16/CUDA:0/eager attention, Transformers 5.12.1. Panels 25/100/50/50; final opened only after development and validation gate pass. Primary factorial 100/50/50 rows; context/KV 10+10 states; Phase-B exact instrumentation pilot one state per model. V33 tests: 6 passed.

|freeze artifact|digest prefix|
|---|---|
|cross_model_rec_conv_v33.freeze.json|aa843d414c7ad586|
|cross_model_rec_conv_v33_adjudication.freeze.json|7d9dd78515dca769|
|cross_model_rec_conv_v33_architecture.freeze.json|604635882bdd8ba3|
|cross_model_rec_conv_v33_context_plan.freeze.json|34a564cc5ef1b4ed|
|cross_model_rec_conv_v33_controls.freeze.json|8ea9be7434e01470|
|cross_model_rec_conv_v33_design.freeze.json|62ff37b529c34323|
|cross_model_rec_conv_v33_factorial_development.freeze.json|3a3e637948f38d2d|
|cross_model_rec_conv_v33_factorial_independent_final.freeze.json|002be259471b5088|
|cross_model_rec_conv_v33_factorial_validation.freeze.json|1fe59f76b02bc216|
|cross_model_rec_conv_v33_final_opening.freeze.json|1b699a0a6ebc7224|
|cross_model_rec_conv_v33_phase_b_instrumentation.freeze.json|0e52dd87f85d0f58|
|cross_model_rec_conv_v33_primary_adjudication.freeze.json|52e36526582587c8|

Integrity index: `results/v33/processed/v33_integrity_index.json`, SHA-256 `8ec97d6e4e87d25ac38bc322395910e80a71b89476c8ac242bb09e79d99184b1`; 74 indexed entries. The index intentionally excludes itself, this manifest and the combined report to avoid circular digests. Model weights are outside Git; checkpoint/tokenizer/config hashes are frozen. Phase-B bidirectional mediation not completed; V33-H/I not established.

Full-suite check: **298 passed, 5 failed, 3 warnings**. All five failures are historical V14/V16/V25/V28/V29 integrity tests that pin an older byte hash of the append-only `reports/FINAL_REPORT.md`; they also failed on the V32 parent and are not V33 causal/runtime failures. The new V33 tests pass 6/6.
