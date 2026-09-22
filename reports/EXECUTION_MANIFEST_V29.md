# Execution Manifest — V29

- `PYTHONPATH=src python -m jclosure.protocol_v29 freeze`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.design_v29`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.forks_v29 calibration`
- `PYTHONPATH=src python -m jclosure.experiments.plan_v29`
- `PYTHONPATH=src python -m jclosure.experiments.amend_v29`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.forks_formal_v29 development`
- `PYTHONPATH=src python -m jclosure.experiments.analyze_v29 development`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.forks_formal_v29 validation`
- `PYTHONPATH=src python -m jclosure.experiments.analyze_v29 validation`
- `PYTHONPATH=src python -m jclosure.experiments.analyze_v29 final_opening`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.secondary_v29 development`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.secondary_v29 validation`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.diagnostics_v29`
- `PYTHONPATH=src python -m jclosure.experiments.realization_plan_v29`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.realization_v29`
- `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m jclosure.experiments.forks_formal_v29 independent_final`
- `PYTHONPATH=src python -m jclosure.experiments.adjudicate_v29`
- `PYTHONPATH=src python -m pytest tests/test_v29_natural_write_content.py -q`
- `PYTHONPATH=src python -m pytest -q`
- `PYTHONPATH=src python -m jclosure.reporting_v29`
- `git commit`
- `git push origin main`

V29 tests: `6 passed in 1.78s`. Full suite: `280 passed, 3 failed, 3 warnings in 52.90s`. Historical cumulative-report hash tests may fail when `FINAL_REPORT.md` is appended; these failures are reported, not suppressed.
