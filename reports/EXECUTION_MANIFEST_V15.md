# V15 execution manifest

Server workspace: `/data/CSK/J-space-project/jstate-closure` on `222.20.126.223`; parent commit `832d1a57f316a59bd961739f7caea4c4e0c702c8`. GPU stages used `HF_HOME=/data/CSK/J-space-project/.hf-cache` and `/home/user/anaconda3/bin/python`. Exact commands from the repository root:

```bash
python -m jclosure.experiments.actuation_v15 --stage freeze
python -m jclosure.experiments.actuation_v15 --stage raw
python scripts/analyze_actuator_transfer_v15.py
python scripts/freeze_v15_splits.py
python -m jclosure.experiments.operator_v15 --stage pilot
python -m jclosure.experiments.precision_v15
python -m jclosure.experiments.operator_v15 --stage linearity
python -m jclosure.experiments.operator_v15 --stage rank
python scripts/analyze_writable_rank_v15.py
python -m jclosure.experiments.basis_v15
python -m jclosure.experiments.repeat_v15
python -m jclosure.experiments.control_v15
python scripts/analyze_control_v15.py
python -m jclosure.experiments.shadow_v15
python scripts/freeze_v15_finalist_decision.py
python scripts/normalize_v15_numeric_json.py
python -m jclosure.reporting_v15
python scripts/build_v15_integrity.py
python scripts/build_complete_version_report.py V15
python -m pytest -q
python -m pytest -q -k 'not test_v14_integrity_manifest'
```

The first transfer summary had an aggregation-only bug and is preserved; `scripts/analyze_actuator_transfer_v15.py` produced a separate frozen correction. The independent confirmation command is absent because `NO_ELIGIBLE_FINALIST`.

Full `pytest -q` has one expected historical test failure: `test_v14_integrity_manifest` compares the old whole-file SHA256 of the cumulative `reports/FINAL_REPORT.md`, which V15 must append. No V14 hash/test/result was modified. The filtered run excludes only this incompatible test and checks the remaining suite plus V15's explicit parent-byte guard.

| Freeze | Digest | File SHA256 |
|---|---|---|
| `artifacts/quantization_aware_actuation_v15.freeze.json` | `8cccf28e4fd09484e1dd2aaf6dcf2abb311695701e439a40eaee6bdee44f7c35` | `02a16fe8dec376a6eaf88cdee72c0bafc80ff65fc1652c14723c6ca3e2fb713f` |
| `artifacts/quantization_aware_actuation_v15_basis.freeze.json` | `b366c35f00f78a10b9fc70fc10b493d6ade00249036f454a7e6d242b9e198c2f` | `2c8dccd3e1ce8c43ef401f4aec2ca139ae3fce3e51593c6431ae24296b6a85ab` |
| `artifacts/quantization_aware_actuation_v15_control.freeze.json` | `d9df5d03b29c23f95c76022aad2d80e1c51c8e4d8342e3502896d1a9b9d79b54` | `c77193aad84c47cca04fcd7535106a1b315d5cfe13074962744f14dcf87292e2` |
| `artifacts/quantization_aware_actuation_v15_control_terminal_analysis.freeze.json` | `93b239663a1556494165fc7cb9bc142a38b859b0e92865f71eb77d4ba72c8b02` | `57789eb85054dd20941227ee27341f5f4da442b55fd317d6b091a498f10e02a7` |
| `artifacts/quantization_aware_actuation_v15_finalist_decision.freeze.json` | `ab14457731da214771b5b18e775084c1e23cbf633f15e67cb441b944351ee5d9` | `b42f17a1be63baaa2ea0224b26855830192563de54f0cf54e0d788d1a201b18f` |
| `artifacts/quantization_aware_actuation_v15_linearity.freeze.json` | `16b3cc515bdf4224a9a9f0bb818f7dca855318c66606f83b6513b385c21bc12c` | `6eea885fb85a969b266ff0d356998fd26f12d37be5d21b4292d625cce825c8b4` |
| `artifacts/quantization_aware_actuation_v15_pilot.freeze.json` | `a6e84657c6add7c19b6dc8223c5a475f7c6860a112e2ff41c93eefb71365091e` | `3d081d5452671eb13e5c674deb3045a5509a1db574e182a1a25821a6cf83a8e6` |
| `artifacts/quantization_aware_actuation_v15_precision.freeze.json` | `b1410a39973502064a9b8b516830e5c62fe9bc33efb902565a89be52ddea51b7` | `99bb877ae1a8e15e21ae75c4b66b1e38bcce55be24833bb29fcf882dfb424393` |
| `artifacts/quantization_aware_actuation_v15_rank.freeze.json` | `1692a35f4de1db7a64d99ef5cecff6cc0c238a827ab777ba204c4e6083ef737a` | `f845b1cf1fcbdd050e28b2feba4ff3825ab1ec3aa726312cc4112ed5fb341570` |
| `artifacts/quantization_aware_actuation_v15_rank_analysis.freeze.json` | `a6e55f93d6d71c617e549d85c1c8820f152f2d42e39f63bdb1d07f402812fde7` | `ce6073480b609774621b65ee1dd8d68b78b267d5f024e01a0709ff44eb6053ca` |
| `artifacts/quantization_aware_actuation_v15_repeat.freeze.json` | `7190eb64963112c79d1791050c779b00c354ad4f5f248a7b1791568b2d8a795c` | `2af303286347093a8ec69cbf78644bdfaf7b97e7f4a1a75dcadacfb18a21537b` |
| `artifacts/quantization_aware_actuation_v15_shadow_accumulation.freeze.json` | `b295fa97cfbdee0a51c564aea4b1c8f105a85d0d6548bc72111eb890d4c8aae8` | `9ae0c3c8e21473381237406a08c95efe6d22b3e2a54087d850d46bbf2028e424` |
| `artifacts/quantization_aware_actuation_v15_splits.freeze.json` | `a534429fce03c842dc4ead9e102e686754b62689e7a3d2111e88b329203f8716` | `2cef4e7bfdb7869056a96bc82e7c1643f40af150539e30647c272fa2200905bb` |
| `artifacts/quantization_aware_actuation_v15_transfer_summary_correction.freeze.json` | `53fb6908ae059344fad21037d2707d0ce861d6e730118f00857a0d4a6e3205dd` | `e002aef201335f09a34d028b99d6cb97442b78af9a6851eb8103e96ab48f6b2d` |

Machine record hashes are in each JSON summary and `results/v15/processed/v15_integrity.json`. Changed files can be enumerated exactly with `git diff --name-only 832d1a57f316a59bd961739f7caea4c4e0c702c8..HEAD` after commit (or `git status --short` before commit).
