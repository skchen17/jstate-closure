# V16 execution manifest

Canonical workspace: `/data/CSK/J-space-project/jstate-closure` on `222.20.126.223`. Parent commit `f1986b1b806112b1c337eb1084b3b82ae11ffb1f`. Runtime: `HF_HOME=/data/CSK/J-space-project/.hf-cache PYTHONPATH=src /home/user/anaconda3/bin/python`; model placement is the frozen V13 dual-GPU loader. Exact repository-root command sequence:

```bash
PYTHONPATH=src python -m jclosure.protocol_v16 freeze
PYTHONPATH=src python -m jclosure.experiments.action_bank_v16 --stage prepare
PYTHONPATH=src python -m jclosure.experiments.action_bank_v16 --stage train --limit 1
PYTHONPATH=src python -m jclosure.experiments.action_bank_v16 --stage amend
PYTHONPATH=src python -m jclosure.experiments.action_bank_v16 --stage train
PYTHONPATH=src python -m jclosure.experiments.action_bank_v16 --stage validation
PYTHONPATH=src python -m jclosure.experiments.analyze_v16 --stage analyze
PYTHONPATH=src python -m jclosure.experiments.realized_v16 --stage run
PYTHONPATH=src python -m jclosure.experiments.composition_v16 --stage run
PYTHONPATH=src python -m jclosure.experiments.sequence_reachability_v16 --stage run
PYTHONPATH=src python -m jclosure.reporting_v16
PYTHONPATH=src python scripts/build_v16_integrity.py
python scripts/build_complete_version_report.py V16
python -m pytest -q -k 'not test_v14_integrity_manifest'
```

The initial Sobol design contained a null vector and stopped before any bank Parquet was written. The original split freeze remains; amendment digest `93b5d89ad9706ac933917e7dc55772776cbfe1dfa0b83edeb5473622ec80ca4d` excludes the null vector and keeps eight nonzero deterministic points. No V1–V15 record was recomputed.

| Freeze | Digest | SHA256 |
|---|---|---|
| `artifacts/nonlinear_finite_causal_action_v16.freeze.json` | `8d64a8093a39401513888fc5681ead119cd983411641b390f1e8cbbc9c8f013a` | `2cb657fd0923ef05e181615e7766bde8f5e6ae86c5eb9dc622dec5365e70fb4b` |
| `artifacts/nonlinear_finite_causal_action_v16_analysis.freeze.json` | `df390b5c258b2705c6c2615e9d6729fb07aa533d9a481ee3bd2a823591237232` | `42a9cc5fe88d963e5c81d30e319be6b053ba3daaeba52d0cae3995f8938b4693` |
| `artifacts/nonlinear_finite_causal_action_v16_bank_source_amendment_1.freeze.json` | `93b5d89ad9706ac933917e7dc55772776cbfe1dfa0b83edeb5473622ec80ca4d` | `4e08b46f2a857164fd0f02af037adbd443f13e339f6f52225989c38c12358876` |
| `artifacts/nonlinear_finite_causal_action_v16_composition.freeze.json` | `c79c01cdfab2c0d23fbea26f84967df33172abbd777ef863b1b9bdc8da72de6b` | `7bf4463af6ef025a8c70908c18bb43fb5f4abb36a7ebe74dec66cd5654e8edc5` |
| `artifacts/nonlinear_finite_causal_action_v16_finalist_decision.freeze.json` | `e2f3ae67bfa1d5e35fabbd7521d1618e2419454fd0c00ac89557e7d134d1e333` | `3f1e9519b5c3fc8f3a4db81991a732fa6d1613c862352453ce906adb78c75c1c` |
| `artifacts/nonlinear_finite_causal_action_v16_realized.freeze.json` | `8767f723cb99e47df8324390befaaebddc202e47e6e2634e52a46ca6a098ec53` | `51c3a918a2f1971a42c7ed5f856604293cfbbf903e3a6d0b43b9971ff698e730` |
| `artifacts/nonlinear_finite_causal_action_v16_sequence_reachability.freeze.json` | `dfd3fbdda58baaa58bcc3ddc4182064760216c03ce85e8016490421349be2862` | `b344d76af61110097c7c50211a34ddee5c29f18a97b276aec2a8b9860dd816c6` |
| `artifacts/nonlinear_finite_causal_action_v16_splits.freeze.json` | `a7d7176534cb3d497c48052c2a5ff67bb2fdcf7feaea3294f45506c16b693859` | `b9ecafa1558527b68a74dfc487c63559890dade0253a86038b7655723c4fca39` |

Train ID hash `d2c6cc991ebe5eb85a47e36895256ad5777141a9f334bacae3779c3136adbe4a`; validation ID hash `9dc52e5e18e2fbec634427fa91935ae192c445de2d80a4c765386e39e551aa43`; independent bank unopened. Machine-record SHA256 values are in the per-stage JSON summaries and `v16_integrity.json`. The filtered test command excludes the pre-existing V14 test that hashes all of mutable cumulative `FINAL_REPORT.md`; no V14 result or test was altered. Source commit for the complete bundle is recorded by `scripts/build_complete_version_report.py` after the first V16 commit.
