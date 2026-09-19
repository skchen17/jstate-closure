# V14 execution and provenance manifest

Server workspace: `/data/CSK/J-space-project/jstate-closure` on `222.20.126.223`. Parent commit: `37df399e4bba8afaca6d99721fe8ceee653e78ee`. V1–V13 tracked guard digest: `8b1829e2f9c4edea0102adc395eb626a93fc0822194d689bf7e67aebc9a1c2d0`. Diagnostic/development ID hashes: `b7efb4bab88dd3c2ab6e89ab0b017136e2cc26efe14ced28b8fb0b9867e97544` / `955ea2e3b3e0f05159838a99e88dfd9b752287a425d86a5374500f438f0547c9`. Independent-final status: `NOT_YET_CREATED_OR_SELECTED`.

Canonical commands (all from repo root with `/home/user/anaconda3/bin/python`; GPU stages use `HF_HOME=/data/CSK/J-space-project/.hf-cache`):

```bash
python -m jclosure.experiments.numerics_v14 --stage freeze
python -m jclosure.experiments.numerics_v14 --stage audit
python -m jclosure.experiments.numerics_v14_extension_1
python -m jclosure.experiments.analyze_numerics_v14
python -m jclosure.experiments.transport_v14 --stage analyze
python -m jclosure.experiments.curvature_v14
python -m jclosure.experiments.closed_loop_v14
python -m jclosure.experiments.numerics_v14_extension_2
python -m jclosure.experiments.channels_v14
python scripts/create_v13_immutable_for_v14.py
python scripts/freeze_v14_splits.py
python scripts/freeze_v14_finalist_decision.py
python scripts/normalize_v14_numeric_json.py
python -m jclosure.reporting_v14
python scripts/build_v14_integrity.py
python scripts/build_complete_version_report.py V14
python -m pytest -q
```

Frozen protocol and amendment hashes:

| freeze                                                                             | SHA256                                                           | digest                                                           |
|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------|:-----------------------------------------------------------------|
| artifacts/finite_causal_control_v14.freeze.json                                    | 8328bf510c87899b27b488a261d108e1d5feb992df0033a69dae3dcfdfedef50 | d268ee6932dbe1bfa1eeecca57c3fc3e97c48fa2793c0564a5e05f6419431cbf |
| artifacts/finite_causal_control_v14_channel_ablation.freeze.json                   | b6743b13aaf6d2205f41d97603b303b1661187bc4e2808a4074e0d7e563c1258 | 05da924b5b71038936b20005831b63145f3b24a3bcc8b282fb6b9531c6033df0 |
| artifacts/finite_causal_control_v14_closed_loop_development.freeze.json            | 21eebcc1a590b68a2c05ff07ec12eb47be5cb775a8a372917d696b704e200a18 | 53cbf8c9b6aa6092330489f4ac635bd55cad939db5cad655063cbd67a9634852 |
| artifacts/finite_causal_control_v14_curvature.freeze.json                          | 992645ee6d331724b0ed31ed94feb09501e9d6f8b0db59f02dfaf174cb6b767a | 4d27b2f63ed064c1ab10870a8eb1457280f0740b9057f5a64930a86996b0efed |
| artifacts/finite_causal_control_v14_finalist_decision.freeze.json                  | 119ec4fe1bad4db1645f4deddd7bb8909a77306f6a4509375a3a97cb20ef68d1 | ffcab29c9b960d1f4dfc490225ced4bc44613ba8841e6f0ca74185015058144c |
| artifacts/finite_causal_control_v14_numerical_partial_fp32_extension_2.freeze.json | 5c3ea5fc485da0f471f1f52e7cf1561b25be21759687472c50ee7fac73cc2a4e | 0125a0ff15632875abd1cfe81805f575762560c7f6e3f7f5e54708447afc940d |
| artifacts/finite_causal_control_v14_numerical_scale_extension_1.freeze.json        | 21b8dac6ab502f2984316866cc02cd4a00716855c36cd81a2058ff29d4923052 | a232ec478862444acd7a29ec2de7b1ff739e00710ac79f0e599f19280bcbc709 |
| artifacts/finite_causal_control_v14_snr_threshold.freeze.json                      | f83106c7c9884e3a716b8fc3dd3928383982bf6d2db8c7841b98a235ef26bdc3 | 0eb484ad5310d349e3ebe34fa39d929c5541b161afda90a4b6accf5825eb32cf |
| artifacts/finite_causal_control_v14_splits.freeze.json                             | bff312bd0211fc9f53a11779357b2d37ae01b9dd2c2567c34a53d7b31deb1bd5 | 07faac4e326aa22d3efbc852edc006f12668a7b7371865ca32461208e2a6c473 |
| artifacts/finite_causal_control_v14_strict_json_correction.freeze.json             | 00a2abdc6570dcbce50f75593cf600e2970ed1b3146bd19838134141d602e720 | 3eaecd2fa9f5218f5ebd1ea96ceff6aecd02f403303f155b72c6bfcabf6c5b97 |
| artifacts/finite_causal_control_v14_transport.freeze.json                          | 4b9f7446dfdacfbd16fb084d7c6f4b2ff4528e7f81318ab974b9b5aebee7b9ae | a62a1063af82b0c8f6766cf3a16553f8364ab292b2ead750872654f4f9e07dd7 |

Machine-record hashes are indexed in `reports/V14_COMPLETE_REPORT.md` for top-level files and in the nested extension summaries for extension records. Exact changed-file list after commit is obtained with `git diff --name-only 37df399e4bba8afaca6d99721fe8ceee653e78ee..HEAD`.
