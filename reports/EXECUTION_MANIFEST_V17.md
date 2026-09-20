# V17 execution manifest

Canonical server: `222.20.126.223:/data/CSK/J-space-project/jstate-closure`. Parent commit `60060cb481b6fe2fe6a9bbf123611d37d4445fea`. Python `/home/user/anaconda3/bin/python`. Commands executed or to be executed for finalization (stage freezes/amendments are additionally enumerated in the strict audit):

- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.protocol_v17 freeze`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.state_sufficiency_v17 prepare`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.state_sufficiency_v17 kernels`
- `OPENBLAS_NUM_THREADS=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.state_sufficiency_v17 ceiling`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.conditional_v17 freeze`
- `OPENBLAS_NUM_THREADS=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.conditional_v17 conditional`
- `OPENBLAS_NUM_THREADS=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.conditional_v17 matched`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.aux_targets_v17 freeze`
- `OPENBLAS_NUM_THREADS=1 PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.experiments.aux_targets_v17 evaluate`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m jclosure.reporting_v17`
- `/home/user/anaconda3/bin/python scripts/repack_v17_records.py`
- `PYTHONPATH=src /home/user/anaconda3/bin/python scripts/build_v17_integrity.py`
- `PYTHONPATH=src /home/user/anaconda3/bin/python -m pytest -q -k 'not test_v14_integrity_manifest and not test_v16_historical_bytes_unchanged_and_manifest'`
- `/home/user/anaconda3/bin/python scripts/build_complete_version_report.py V17`

Primary files: `configs/interventional_state_sufficiency_v17.yaml`, `src/jclosure/protocol_v17.py`, three V17 experiment modules, `src/jclosure/reporting_v17.py`, `scripts/repack_v17_records.py`, `scripts/build_v17_integrity.py`, `tests/test_v17.py`, V17 freeze manifests, `results/v17/processed/*`, `reports/*_V17.md`, and append-only `reports/FINAL_REPORT.md`. Exact file hashes are in `results/v17/processed/v17_integrity.json` and the single-file `reports/V17_COMPLETE_REPORT.md`.
